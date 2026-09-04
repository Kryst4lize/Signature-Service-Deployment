"""Image and PDF loading, plus base64 encoding for API responses.

Merged from the former `services/preprocessing.py` and `services/image_utils.py`.
"""

import base64
import io
import logging

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_bytes

    PDF_SUPPORT = True
except ImportError:  # pragma: no cover - poppler-utils missing
    PDF_SUPPORT = False

YOLO_SIZE = (640, 640)   # YOLOv8 input  - used by /verify-document
MODEL_SIZE = (224, 224)  # CycleGAN / ResNet50 / VGG16 - used by /register-signature


# ── Loading ───────────────────────────────────────────────────────────────────


def load_image_as_tensor(
    data: bytes,
    filename: str,
    size: tuple[int, int] = YOLO_SIZE,
) -> list[np.ndarray]:
    """Load an uploaded file (image or PDF) into a list of float32 tensors
    shaped [1, 3, H, W], one per page.
    """
    if filename.lower().endswith(".pdf"):
        return _pdf_to_tensors(data, size)
    return [_bytes_to_tensor(data, size)]


def _pdf_to_tensors(data: bytes, size: tuple[int, int]) -> list[np.ndarray]:
    if not PDF_SUPPORT:
        raise RuntimeError("pdf2image not available - install poppler-utils")
    pages = convert_from_bytes(data, dpi=200)
    logger.info("PDF decoded to %d page(s)", len(pages))
    return [_pil_to_tensor(page, size) for page in pages]


def _bytes_to_tensor(data: bytes, size: tuple[int, int]) -> np.ndarray:
    return _pil_to_tensor(Image.open(io.BytesIO(data)).convert("RGB"), size)


def _pil_to_tensor(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(img.resize(size, Image.LANCZOS), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]


# ── Encoding ──────────────────────────────────────────────────────────────────


def tensor_to_pil(tensor: np.ndarray) -> Image.Image:
    """[1, 3, H, W] float32 in [0, 1] -> PIL RGB."""
    arr = tensor.squeeze(0).transpose(1, 2, 0)
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), "RGB")


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def tensor_to_b64(tensor: np.ndarray, fmt: str = "PNG") -> str:
    return pil_to_b64(tensor_to_pil(tensor), fmt)


def draw_bbox_on_tensor(
    page_tensor: np.ndarray,
    bbox: list[float],
    color: str = "#FF2D2D",
    width: int = 3,
) -> str:
    """Draw a rectangle on the page tensor and return it as base64 PNG."""
    img = tensor_to_pil(page_tensor)
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = (int(v) for v in bbox)
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    return pil_to_b64(img)
