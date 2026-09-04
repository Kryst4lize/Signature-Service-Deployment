import logging
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile 
from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.db import Item
from app.services.image_utils import draw_bbox_on_tensor, tensor_to_b64
from app.services.preprocessing import load_image_as_tensor
from app.services.triton import TritonService, get_triton_service

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()


# ══ GET /signatures ═══════════════════════════════════════════════════════════

@router.get("/signatures")
async def list_signatures(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Return all registered signature entries (no vectors)."""
    logger.info("GET /signatures called with skip=%d, limit=%d", skip, limit)
    total_result = await db.execute(select(func.count()).select_from(Item))
    total = total_result.scalar()
    logger.info("Total signature entries found in DB: %d", total)

    result = await db.execute(
        select(
            Item.id,
            Item.username,
            Item.user_created_date,
            Item.user_modified_date,
        )
        .offset(skip)
        .limit(limit)
        .order_by(Item.user_created_date.desc())
    )
    rows = result.mappings().all()
    logger.info("Returning %d signature entries", len(rows))
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "items": [dict(r) for r in rows],
    }


# ══ DELETE /signatures/{id} ════════════════════════════════════════════════════

@router.delete("/signatures/{item_id}", status_code=204)
async def delete_signature(item_id: int, db: AsyncSession = Depends(get_db)):
    logger.info("DELETE /signatures/%d called", item_id)
    result = await db.execute(select(Item).where(Item.id == item_id))
    item = result.scalar_one_or_none()
    if item is None:
        logger.info("Signature ID %d not found in DB", item_id)
        raise HTTPException(status_code=404, detail="Signature not found")
    
    await db.delete(item)
    logger.info("Successfully deleted signature ID %d", item_id)


# ══ POST /register-signature ══════════════════════════════════════════════════

@router.post("/register-signature", status_code=201)
async def register_signature(
    username: str = Form(...),
    file: UploadFile = File(..., description="Clean signature image (jpg/png)"),
    db: AsyncSession = Depends(get_db),
    triton: TritonService = Depends(get_triton_service),
):
    """
    Pipeline: upload → CycleGAN denoise → ResNet50 + VGG16 extract → INSERT
    YOLOv8 detection is skipped: the uploaded file is already an isolated signature.
    """
    logger.info("POST /register-signature called for username: %s", username)
    logger.info("Reading file: %s", file.filename)
    raw = await file.read()
    
    try:
        logger.info("Converting uploaded file to tensor (size: 224x224)")
        tensors = load_image_as_tensor(raw, file.filename, size=(224, 224))
    except Exception as exc:
        logger.error("Could not decode image: %s", exc)
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    image = tensors[0]

    try:
        logger.info("Calling Triton denoise (CycleGAN)")
        clean = await triton.denoise(image)
    except Exception as exc:
        logger.error("Triton CycleGAN error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Triton CycleGAN error: {exc}")

    try:
        logger.info("Calling Triton extract_features (ResNet50 + VGG16)")
        resnet_vec, vgg_vec = await triton.extract_features(clean)
    except Exception as exc:
        logger.error("Triton extractor error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Triton extractor error: {exc}")

    logger.info("Instantiating new Item for DB insert")
    item = Item(
        username=username,
        resnet50_vector=resnet_vec.tolist(),
        vgg16_vector=vgg_vec.tolist(),
    )
    db.add(item)
    await db.flush()
    await db.refresh(item)
    
    logger.info("Successfully registered signature for %s (ID: %d)", username, item.id)

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
              → ResNet50+VGG16 extract → pgvector L2 search → result + images
    """
    logger.info("POST /verify-document called for file: %s", file.filename)
    raw = await file.read()
    try:
        logger.info("Loading document into pages/tensors")
        pages = load_image_as_tensor(raw, file.filename)
        logger.info("Document loaded successfully. Total pages: %d", len(pages))
    except Exception as exc:
        logger.error("Could not decode file: %s", exc)
        raise HTTPException(status_code=400, detail=f"Could not decode file: {exc}")

    results = []

    for page_idx, page_tensor in enumerate(pages):
        logger.info("--- Processing page %d of %d ---", page_idx + 1, len(pages))
        entry: dict = {"page": page_idx + 1}

        # ── Step 1: YOLOv8 detect ────────────────────────────────────────────
        try:
            logger.info("Step 1: Running YOLOv8 detection on page %d", page_idx + 1)
            crop, bbox = await triton.detect_signature(page_tensor)
        except Exception as exc:
            logger.error("YOLOv8 detection failed on page %d: %s", page_idx + 1, exc)
            entry.update({"status": "no_signature", "detail": str(exc)})
            results.append(entry)
            continue

        if crop is None or crop.size == 0:
            logger.info("No signature detected on page %d", page_idx + 1)
            entry.update({"status": "no_signature"})
            results.append(entry)
            continue

        logger.info("Signature detected on page %d. BBox: %s", page_idx + 1, bbox)
        # Annotate original page with red bbox
        logger.info("Annotating original page with bounding box")
        page_annotated_b64 = (
            draw_bbox_on_tensor(page_tensor, bbox) if bbox else tensor_to_b64(page_tensor)
        )
        crop_before_b64 = tensor_to_b64(crop)
        entry["bbox"] = bbox
        entry["page_annotated"] = page_annotated_b64
        entry["crop_before"] = crop_before_b64

        # ── Step 2: CycleGAN denoise ─────────────────────────────────────────
        try:
            logger.info("Step 2: Denoising cropped signature (CycleGAN)")
            clean = await triton.denoise(crop)
        except Exception as exc:
            logger.error("Triton CycleGAN error on page %d: %s", page_idx + 1, exc)
            raise HTTPException(status_code=502, detail=f"Triton CycleGAN error: {exc}")

        entry["crop_after"] = tensor_to_b64(clean)

        # ── Step 3: feature extraction ───────────────────────────────────────
        try:
            logger.info("Step 3: Extracting features (ResNet50 + VGG16)")
            resnet_vec, vgg_vec = await triton.extract_features(clean)
        except Exception as exc:
            logger.error("Triton extractor error on page %d: %s", page_idx + 1, exc)
            raise HTTPException(status_code=502, detail=f"Triton extractor error: {exc}")

        # ── Step 4: pgvector L2 nearest neighbour ────────────────────────────
        logger.info("Step 4: Querying DB for L2 nearest neighbour")
        row = await db.execute(
            text("""
                SELECT id, username,
                       (resnet50_vector <-> CAST(:rv AS vector)) AS d_resnet,
                       (vgg16_vector    <-> CAST(:vv AS vector)) AS d_vgg,
                       (
                           (resnet50_vector <-> CAST(:rv AS vector)) +
                           (vgg16_vector    <-> CAST(:vv AS vector))
                       ) / 2 AS avg_distance
                FROM items
                ORDER BY avg_distance ASC
                LIMIT 1
            """),
            {
                "rv": str(resnet_vec.tolist()), 
                "vv": str(vgg_vec.tolist())
            },
        )
        match = row.mappings().first()

        if match is None:
            logger.info("No matching signature found in DB for page %d", page_idx + 1)
            entry.update({"status": "no_match_in_db"})
        else:
            avg_dist = float(match["avg_distance"])
            is_matched = avg_dist < 0.5
            logger.info(
                "DB search complete for page %d. Best match: %s, Avg Dist: %f, Matched: %s", 
                page_idx + 1, match["username"], avg_dist, is_matched
            )
            entry.update({
                "status": "matched",
                "username": match["username"],
                "avg_distance": round(avg_dist, 6),
                "resnet_distance": round(float(match["d_resnet"]), 6),
                "vgg_distance": round(float(match["d_vgg"]), 6),
                "matched": is_matched,
            })

        results.append(entry)

    logger.info("Finished verifying document. Returning results for %d pages.", len(pages))
    return {"total_pages": len(pages), "results": results}