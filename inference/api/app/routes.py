import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import Item, get_db
from app.images import draw_bbox, load_pages, pil_to_b64, pil_to_tensor, tensor_to_pil
from app.triton import MODEL_SIZE, YOLO_SIZE, TritonService, get_triton_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Annotated pages are echoed back as base64; cap their long edge so a 200-dpi A4
# scan does not turn into several MB of JSON per page.
PREVIEW_MAX_EDGE = 1024

_NEAREST_SQL = text(
    """
    SELECT id,
           username,
           (resnet50_vector <=> CAST(:rv AS vector)) AS d_resnet,
           (vgg16_vector    <=> CAST(:vv AS vector)) AS d_vgg,
           (
               (resnet50_vector <=> CAST(:rv AS vector)) +
               (vgg16_vector    <=> CAST(:vv AS vector))
           ) / 2 AS avg_distance
    FROM items
    WHERE resnet50_vector IS NOT NULL
      AND vgg16_vector    IS NOT NULL
    ORDER BY avg_distance ASC
    LIMIT 1
    """
)


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload, refusing anything over the configured size cap."""
    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_bytes} byte limit",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    return raw


def _decode(raw: bytes, filename: str) -> list[Image.Image]:
    try:
        return load_pages(raw, filename, settings.max_pdf_pages)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Unsupported or corrupt image file"
        ) from None
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode file: {exc}"
        ) from exc


def _preview(img: Image.Image) -> tuple[Image.Image, float]:
    """Downscale for the base64 preview echoed back in the response.

    Returns the image and the scale factor applied, so callers can map
    full-resolution coordinates onto the preview.
    """
    longest = max(img.size)
    if longest <= PREVIEW_MAX_EDGE:
        return img, 1.0
    scale = PREVIEW_MAX_EDGE / longest
    resized = img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.LANCZOS,
    )
    return resized, scale


# ══ GET /signatures ═══════════════════════════════════════════════════════════


@router.get("/signatures")
async def list_signatures(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Registered signature entries, newest first. Vectors are not returned."""
    total = (await db.execute(select(func.count()).select_from(Item))).scalar()
    rows = (
        await db.execute(
            select(
                Item.id,
                Item.username,
                Item.user_created_date,
                Item.user_modified_date,
            )
            .order_by(Item.user_created_date.desc())
            .offset(skip)
            .limit(limit)
        )
    ).mappings().all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "items": [dict(r) for r in rows],
    }


# ══ DELETE /signatures/{id} ═══════════════════════════════════════════════════


@router.delete("/signatures/{item_id}", status_code=204)
async def delete_signature(item_id: int, db: AsyncSession = Depends(get_db)):
    item = (await db.execute(select(Item).where(Item.id == item_id))).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Signature not found")
    await db.delete(item)


# ══ POST /register-signature ══════════════════════════════════════════════════


@router.post("/register-signature", status_code=201)
async def register_signature(
    username: str = Form(..., min_length=1, max_length=50),
    file: UploadFile = File(..., description="Clean signature image (jpg/png)"),
    db: AsyncSession = Depends(get_db),
    triton: TritonService = Depends(get_triton_service),
):
    """upload -> CycleGAN denoise -> ResNet50 + VGG16 -> INSERT.

    Detection is skipped: the upload is assumed to be an isolated signature.
    """
    raw = await _read_upload(file)
    pages = _decode(raw, file.filename or "")
    tensor = pil_to_tensor(pages[0], (MODEL_SIZE, MODEL_SIZE))

    try:
        clean = await triton.denoise(tensor)
        resnet_vec, vgg_vec = await triton.extract_features(clean)
    except Exception as exc:
        logger.exception("Triton inference failed during registration")
        raise HTTPException(
            status_code=502, detail=f"Triton inference error: {exc}"
        ) from exc

    item = Item(
        username=username,
        resnet50_vector=resnet_vec.tolist(),
        vgg16_vector=vgg_vec.tolist(),
    )
    db.add(item)
    await db.flush()
    await db.refresh(item)
    logger.info("Registered signature %d for %s", item.id, username)

    return {
        "id": item.id,
        "username": item.username,
        "user_created_date": item.user_created_date,
    }


# ══ POST /verify-document ═════════════════════════════════════════════════════


@router.post("/verify-document")
async def verify_document(
    file: UploadFile = File(..., description="Document PDF or image"),
    db: AsyncSession = Depends(get_db),
    triton: TritonService = Depends(get_triton_service),
):
    """upload -> pages -> YOLOv8 detect -> crop at full resolution
    -> CycleGAN denoise -> ResNet50 + VGG16 -> pgvector cosine search.
    """
    raw = await _read_upload(file)
    pages = _decode(raw, file.filename or "")
    logger.info("Verifying %s (%d page(s))", file.filename, len(pages))

    results = []
    for page_idx, page in enumerate(pages):
        entry: dict = {"page": page_idx + 1}

        # ── Detect on a 640x640 view of the page ──────────────────────────────
        try:
            detection = await triton.detect_signature(
                pil_to_tensor(page, (YOLO_SIZE, YOLO_SIZE))
            )
        except Exception as exc:
            logger.exception("Detection failed on page %d", page_idx + 1)
            entry.update({"status": "no_signature", "detail": str(exc)})
            results.append(entry)
            continue

        if detection is None:
            entry["status"] = "no_signature"
            results.append(entry)
            continue

        bbox_norm, confidence = detection

        # ── Crop from the ORIGINAL page, not from the 640x640 detector input ──
        # Cropping the detector input would bake its downsampling into the
        # embedding, so a verified crop would be visibly blurrier than the sharp
        # image the same signature was enrolled from.
        width, height = page.size
        box = (
            int(bbox_norm[0] * width),
            int(bbox_norm[1] * height),
            int(bbox_norm[2] * width),
            int(bbox_norm[3] * height),
        )
        crop = page.crop(box)
        if crop.width < 2 or crop.height < 2:
            entry["status"] = "no_signature"
            results.append(entry)
            continue

        page_preview, scale = _preview(page)
        entry["bbox"] = list(box)
        entry["confidence"] = round(confidence, 4)
        entry["page_annotated"] = draw_bbox(page_preview, [c * scale for c in box])
        entry["crop_before"] = pil_to_b64(_preview(crop)[0])

        # ── Denoise + embed ───────────────────────────────────────────────────
        try:
            clean = await triton.denoise(pil_to_tensor(crop, (MODEL_SIZE, MODEL_SIZE)))
            resnet_vec, vgg_vec = await triton.extract_features(clean)
        except Exception as exc:
            logger.exception("Triton inference failed on page %d", page_idx + 1)
            raise HTTPException(
                status_code=502, detail=f"Triton inference error: {exc}"
            ) from exc

        entry["crop_after"] = pil_to_b64(tensor_to_pil(clean))

        # ── Nearest neighbour by cosine distance ──────────────────────────────
        match = (
            await db.execute(
                _NEAREST_SQL,
                {"rv": str(resnet_vec.tolist()), "vv": str(vgg_vec.tolist())},
            )
        ).mappings().first()

        if match is None:
            entry["status"] = "no_match_in_db"
        else:
            avg = float(match["avg_distance"])
            entry.update(
                {
                    "status": "matched",
                    "username": match["username"],
                    "avg_distance": round(avg, 6),
                    "resnet_distance": round(float(match["d_resnet"]), 6),
                    "vgg_distance": round(float(match["d_vgg"]), 6),
                    "matched": avg < settings.match_threshold,
                }
            )

        results.append(entry)

    return {"total_pages": len(pages), "results": results}
