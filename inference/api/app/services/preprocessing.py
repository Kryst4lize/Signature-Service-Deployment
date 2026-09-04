import io
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

YOLO_SIZE  = (640, 640)   # YOLOv8 input  – used by /verify-document
MODEL_SIZE = (224, 224)   # CycleGAN / ResNet50 / VGG16 – used by /register-signature


def load_image_as_tensor(
    data: bytes,
    filename: str,
    size: tuple[int, int] = YOLO_SIZE,
) -> list[np.ndarray]:
    """
    Load an uploaded file (image or PDF) and return a list of
    float32 tensors shaped [1, 3, H, W], one per page/image.

    size = YOLO_SIZE  (640) for /verify-document  (fed into YOLOv8)
    size = MODEL_SIZE (224) for /register-signature (fed directly into CycleGAN)
    """
    logger.info("Entering load_image_as_tensor for file: %s with target size: %s", filename, size)
    if filename.lower().endswith(".pdf"):
        logger.info("File identified as PDF.")
        return _pdf_to_tensors(data, size)
    
    logger.info("File identified as standard image.")
    return [_bytes_to_tensor(data, size)]


def _pdf_to_tensors(data: bytes, size: tuple[int, int]) -> list[np.ndarray]:
    logger.info("Entering _pdf_to_tensors")
    if not PDF_SUPPORT:
        logger.error("pdf2image not available – install poppler-utils")
        raise RuntimeError("pdf2image not available – install poppler-utils")
    
    logger.info("Converting PDF bytes to PIL images at 200 dpi")
    pages = convert_from_bytes(data, dpi=200)
    logger.info("PDF conversion yielded %d pages", len(pages))
    
    return [_pil_to_tensor(page, size) for page in pages]


def _bytes_to_tensor(data: bytes, size: tuple[int, int]) -> np.ndarray:
    logger.info("Entering _bytes_to_tensor")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    logger.info("Image successfully loaded from bytes and converted to RGB")
    return _pil_to_tensor(img, size)


def _pil_to_tensor(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    logger.info("Entering _pil_to_tensor to resize image to %s", size)
    img = img.resize(size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, 3]
    arr = arr.transpose(2, 0, 1)                    # [3, H, W]
    tensor = np.expand_dims(arr, axis=0)            # [1, 3, H, W]
    logger.info("Image converted to tensor of shape %s", tensor.shape)
    return tensor