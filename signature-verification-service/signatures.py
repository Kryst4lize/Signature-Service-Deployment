from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pgvector.asyncpg import register_vector
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.db import Item
from app.services.preprocessing import load_image_as_tensor
from app.services.triton import TritonService, get_triton_service

router = APIRouter()


# ══ POST /register-signature ═══════════════════════════════════════════════════

@router.post("/register-signature", status_code=201)
async def register_signature(
    username: str = Form(...),
    file: UploadFile = File(..., description="Clean signature image (jpg/png)"),
    db: AsyncSession = Depends(get_db),
    triton: TritonService = Depends(get_triton_service),
):
    """
    Pipeline: upload → denoise (CycleGAN) → extract vectors (ResNet50 + VGG16)
              → INSERT into items
    Note: YOLOv8 detection is skipped here because the uploaded file is
          already an isolated signature crop.
    """
    raw = await file.read()

    try:
        tensors = load_image_as_tensor(raw, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    image = tensors[0]  # single image expected for registration

    # Step 1 – CycleGAN denoising
    try:
        clean = await triton.denoise(image)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Triton CycleGAN error: {exc}")

    # Step 2 – parallel feature extraction
    try:
        resnet_vec, vgg_vec = await triton.extract_features(clean)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Triton extractor error: {exc}")

    # Step 3 – persist
    item = Item(
        username=username,
        resnet50_vector=resnet_vec.tolist(),
        vgg16_vector=vgg_vec.tolist(),
    )
    db.add(item)
    await db.flush()
    await db.refresh(item)

    return {
        "id": item.id,
        "username": item.username,
        "user_created_date": item.user_created_date,
    }


# ══ POST /verify-document ══════════════════════════════════════════════════════

@router.post("/verify-document")
async def verify_document(
    file: UploadFile = File(..., description="Document PDF or image"),
    db: AsyncSession = Depends(get_db),
    triton: TritonService = Depends(get_triton_service),
):
    """
    Pipeline: upload → (PDF→pages) → YOLOv8 detect → CycleGAN denoise
              → ResNet50 + VGG16 extract → pgvector L2 search → result
    """
    raw = await file.read()

    try:
        pages = load_image_as_tensor(raw, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode file: {exc}")

    results = []

    for page_idx, page_tensor in enumerate(pages):

        # Step 1 – YOLOv8: detect & crop signature
        try:
            crop = await triton.detect_signature(page_tensor)
        except Exception as exc:
            results.append({"page": page_idx + 1, "status": "no_signature", "detail": str(exc)})
            continue

        if crop is None or crop.size == 0:
            results.append({"page": page_idx + 1, "status": "no_signature"})
            continue

        # Step 2 – CycleGAN: remove noise
        try:
            clean = await triton.denoise(crop)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Triton CycleGAN error: {exc}")

        # Step 3 – parallel feature extraction
        try:
            resnet_vec, vgg_vec = await triton.extract_features(clean)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Triton extractor error: {exc}")

        # Step 4 – pgvector L2 nearest-neighbour (average of both distances)
        row = await db.execute(
            text("""
                SELECT id, username,
                       (resnet50_vector <-> :rv) AS d_resnet,
                       (vgg16_vector    <-> :vv) AS d_vgg,
                       (
                           (resnet50_vector <-> :rv) +
                           (vgg16_vector    <-> :vv)
                       ) / 2 AS avg_distance
                FROM items
                ORDER BY avg_distance ASC
                LIMIT 1
            """),
            {
                "rv": resnet_vec.tolist(),
                "vv": vgg_vec.tolist(),
            },
        )
        match = row.mappings().first()

        if match is None:
            results.append({"page": page_idx + 1, "status": "no_match_in_db"})
            continue

        results.append({
            "page": page_idx + 1,
            "status": "matched",
            "username": match["username"],
            "avg_distance": round(float(match["avg_distance"]), 6),
            "resnet_distance": round(float(match["d_resnet"]), 6),
            "vgg_distance": round(float(match["d_vgg"]), 6),
            "matched": float(match["avg_distance"]) < 0.5,  # tune threshold as needed
        })

    return {"total_pages": len(pages), "results": results}
