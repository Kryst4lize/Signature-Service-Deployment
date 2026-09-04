import asyncio
import logging
from functools import lru_cache

import numpy as np
import tritonclient.http.aio as httpclient
from PIL import Image as PILImage

from app.config import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── size constants ─────────────────────────────────────────────────────────────
YOLO_SIZE    = 640   # YOLOv8 input
MODEL_SIZE   = 224   # CycleGAN / ResNet50 / VGG16 input
CONF_THRESH  = 0.5  # YOLOv8 confidence threshold


def _resize_crop_to_model(crop_arr: np.ndarray) -> np.ndarray:
    """[3, H, W] float32 → [1, 3, 224, 224] float32"""
    logger.info("Resizing cropped array to model size (%dx%d)", MODEL_SIZE, MODEL_SIZE)
    h, w = crop_arr.shape[1], crop_arr.shape[2]
    pil = PILImage.fromarray(
        (crop_arr.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    )
    pil = pil.resize((MODEL_SIZE, MODEL_SIZE), PILImage.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 255.0   # [H, W, 3]
    arr = arr.transpose(2, 0, 1)[np.newaxis]        # [1, 3, 224, 224]
    logger.info("Resized crop tensor shape: %s", arr.shape)
    return arr


class TritonService:

    def __init__(self) -> None:
        self._client: httpclient.InferenceServerClient | None = None
        logger.info("TritonService initialized")

    async def connect(self) -> None:
        logger.info("Connecting to Triton server at URL: %s", settings.triton_http_url)
        self._client = httpclient.InferenceServerClient(
            url=settings.triton_http_url, verbose=False
        )
        logger.info("Connected to Triton server")

    async def close(self) -> None:
        if self._client:
            logger.info("Closing Triton client connection")
            await self._client.close()
            logger.info("Triton client connection closed")

    async def _infer(
        self,
        model_name: str,
        inputs: list[httpclient.InferInput],
        output_names: list[str],
    ) -> dict[str, np.ndarray]:
        logger.info("Executing inference request on model: %s", model_name)
        outputs = [httpclient.InferRequestedOutput(n) for n in output_names]
        result  = await self._client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )
        logger.info("Inference completed on model: %s", model_name)
        return {n: result.as_numpy(n) for n in output_names}

    # ── Model 1: yolov8s ──────────────────────────────────────────────────────
    async def detect_signature(
        self, image: np.ndarray  # [1, 3, 640, 640]
    ) -> tuple[np.ndarray | None, list[float] | None]:
        """
        Returns (crop [1,3,224,224], bbox [x1,y1,x2,y2]) or (None, None).
        output0 shape from Triton: [5, N]  →  rows = [cx, cy, w, h, conf]
        """
        logger.info("Starting detect_signature with model 'yolov8s'")
        inp = httpclient.InferInput("images", image.shape, "FP32")
        inp.set_data_from_numpy(image)
        result = await self._infer("yolov8s", [inp], ["output0"])

        raw = result["output0"]           # may be [5, N] or [1, 5, N]
        if raw.ndim == 3:
            raw = raw[0]                  # → [5, N]

        if raw.shape[1] == 0:
            logger.info("detect_signature: No predictions returned from yolov8s")
            return None, None

        confidences = raw[4, :]
        best        = int(np.argmax(confidences))

        logger.info("Best YOLOv8 confidence score: %f", confidences[best])
        if confidences[best] < CONF_THRESH:
            logger.info("Confidence %f is below threshold %f. Discarding.", confidences[best], CONF_THRESH)
            return None, None

        cx, cy, w, h = float(raw[0, best]), float(raw[1, best]), \
                       float(raw[2, best]), float(raw[3, best])

        img_h, img_w = image.shape[2], image.shape[3]   # 640, 640
        x1 = max(0,     int(cx - w / 2))
        y1 = max(0,     int(cy - h / 2))
        x2 = min(img_w, int(cx + w / 2))
        y2 = min(img_h, int(cy + h / 2))

        if x2 <= x1 or y2 <= y1:
            logger.info("Invalid bounding box generated: [%d, %d, %d, %d]", x1, y1, x2, y2)
            return None, None

        bbox     = [x1, y1, x2, y2]
        crop_arr = image[0, :, y1:y2, x1:x2]   # [3, H', W']
        logger.info("Bounding box selected: %s. Proceeding to resize crop.", bbox)
        crop_224 = _resize_crop_to_model(crop_arr)

        return crop_224, bbox

    # ── Model 2: latest_net_G_B (CycleGAN) ───────────────────────────────────
    async def denoise(self, crop: np.ndarray) -> np.ndarray:
        """Input/output: [1, 3, 224, 224]"""
        logger.info("Starting denoise with model 'latest_net_G_B'")
        inp = httpclient.InferInput("input", crop.shape, "FP32")
        inp.set_data_from_numpy(crop)
        result = await self._infer("latest_net_G_B", [inp], ["output"])
        logger.info("Denoise completed")
        return result["output"]

    # ── Model 3 + 4: resnet50_extractor & vgg16_extractor ────────────────────
    async def extract_features(
        self, clean_image: np.ndarray  # [1, 3, 224, 224]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parallel inference. Returns (resnet_vec, vgg_vec) each [4096]."""
        logger.info("Starting parallel feature extraction (ResNet50 & VGG16)")

        # 1. ResNet50 expects channels first (NCHW: [1, 3, 224, 224])
        # Send the exact same NCHW tensor to both models
        resnet_inp = httpclient.InferInput("input_layer_1", clean_image.shape, "FP32")
        resnet_inp.set_data_from_numpy(clean_image)

        vgg_inp = httpclient.InferInput("input_layer", clean_image.shape, "FP32")
        vgg_inp.set_data_from_numpy(clean_image)

        logger.info("Awaiting resnet50_extractor and vgg16_extractor tasks")
        resnet_res, vgg_res = await asyncio.gather(
            self._infer("resnet50_extractor", [resnet_inp], ["fc1"]),
            self._infer("vgg16_extractor",    [vgg_inp],    ["fc1"]),
        )
        logger.info("Feature extraction successfully completed")

        return (
            resnet_res["fc1"].flatten().astype(np.float32),
            vgg_res["fc1"].flatten().astype(np.float32),
        )


@lru_cache(maxsize=1)
def get_triton_service() -> TritonService:
    logger.info("Fetching TritonService singleton instance")
    return TritonService()