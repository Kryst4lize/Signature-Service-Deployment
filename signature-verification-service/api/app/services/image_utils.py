import base64
import io

import numpy as np
from PIL import Image, ImageDraw


def tensor_to_pil(tensor: np.ndarray) -> Image.Image:
    """Convert [1, 3, H, W] float32 0-1 tensor to PIL RGB image."""
    arr = tensor.squeeze(0).transpose(1, 2, 0)           # [H, W, 3]
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()



def tensor_to_b64(tensor: np.ndarray, fmt: str = "PNG") -> str:
    return pil_to_b64(tensor_to_pil(tensor), fmt)


def draw_bbox_on_tensor(
    page_tensor: np.ndarray,       # [1, 3, H, W]
    bbox: list[float],             # [x1, y1, x2, y2] in pixel coords
    color: str = "#FF2D2D",
    width: int = 3,
) -> str:
    """Draw red rectangle on the page and return as base64 PNG."""
    img  = tensor_to_pil(page_tensor)
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    return pil_to_b64(img)