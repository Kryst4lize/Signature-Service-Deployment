"""Image and PDF I/O.

Everything crossing this module's boundary as a tensor is

    float32, [1, 3, H, W], values in [0, 1], RGB channel order.

Model-specific rescaling lives in `triton.py`, next to the model that needs it.
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

PDF_DPI = 200


# ── Loading ───────────────────────────────────────────────────────────────────


def load_pages(data: bytes, filename: str, max_pages: int) -> list[Image.Image]:
    """Decode an upload into full-resolution RGB pages.

    Resizing is deliberately not done here. /verify-document needs the original
    resolution to crop a sharp signature after detection; downscaling up front
    would discard that detail irrecoverably.
    """
    if filename.lower().endswith(".pdf"):
        if not PDF_SUPPORT:
            raise RuntimeError("pdf2image not available - install poppler-utils")
        pages = convert_from_bytes(data, dpi=PDF_DPI, first_page=1, last_page=max_pages)
        logger.info("PDF decoded to %d page(s) at %d dpi", len(pages), PDF_DPI)
        return [p.convert("RGB") for p in pages]

    return [Image.open(io.BytesIO(data)).convert("RGB")]


def pil_to_tensor(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """PIL RGB -> [1, 3, H, W] float32 in [0, 1]."""
    arr = np.asarray(img.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
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


def draw_bbox(
    img: Image.Image,
    bbox: list[float],
    color: str = "#FF2D2D",
    width: int = 3,
) -> str:
    """Draw a rectangle on a copy of `img`, return it as base64 PNG."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = (int(v) for v in bbox)
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    return pil_to_b64(out)
