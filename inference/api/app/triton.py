"""Triton client.

Every tensor crossing this module's public boundary is

    float32, [1, 3, H, W], values in [0, 1], RGB channel order.

Each model wants something different, and each of those conversions is applied
here, immediately next to the inference call that needs it:

    yolov8s             [0, 1] RGB          (ultralytics convention)
    latest_net_G_B      [-1, 1] RGB         (CycleGAN Normalize(0.5, 0.5);
                                             the generator ends in Tanh)
    resnet50_extractor  Caffe BGR           (x*255, RGB->BGR, ImageNet mean
    vgg16_extractor                          subtracted - what Keras
                                             preprocess_input(mode="caffe") does)

Getting any of these wrong is silent: the models still return well-formed
tensors of the right shape, they are just computed on out-of-distribution input.
"""

import asyncio
import logging
from functools import lru_cache

import numpy as np
import tritonclient.http.aio as httpclient

from app.config import settings

logger = logging.getLogger(__name__)

YOLO_SIZE = 640  # yolov8s input
MODEL_SIZE = 224  # CycleGAN / ResNet50 / VGG16 input

# keras.applications.imagenet_utils.preprocess_input(mode="caffe"), BGR order.
_CAFFE_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 3, 1, 1)


def to_cyclegan(x: np.ndarray) -> np.ndarray:
    """[0, 1] -> [-1, 1]."""
    return x * 2.0 - 1.0


def from_cyclegan(x: np.ndarray) -> np.ndarray:
    """[-1, 1] Tanh output -> [0, 1]."""
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def to_caffe(x: np.ndarray) -> np.ndarray:
    """[0, 1] RGB -> [0, 255] BGR with the ImageNet mean subtracted."""
    bgr = x[:, ::-1, :, :] * 255.0
    return (bgr - _CAFFE_MEAN_BGR).astype(np.float32)


def l2_normalise(vec: np.ndarray) -> np.ndarray:
    """Unit-length embedding, matching what the training and evaluation code
    stores. Without this the cosine/L2 thresholds are meaningless."""
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


class TritonService:
    def __init__(self) -> None:
        self._client: httpclient.InferenceServerClient | None = None

    async def connect(self) -> None:
        self._client = httpclient.InferenceServerClient(url=settings.triton_http_url, verbose=False)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def _infer(
        self,
        model_name: str,
        input_name: str,
        tensor: np.ndarray,
        output_names: list[str],
    ) -> dict[str, np.ndarray]:
        if self._client is None:
            raise RuntimeError("Triton client is not connected")
        inp = httpclient.InferInput(input_name, tensor.shape, "FP32")
        inp.set_data_from_numpy(np.ascontiguousarray(tensor, dtype=np.float32))
        result = await self._client.infer(
            model_name=model_name,
            inputs=[inp],
            outputs=[httpclient.InferRequestedOutput(n) for n in output_names],
        )
        return {n: result.as_numpy(n) for n in output_names}

    # ── yolov8s ───────────────────────────────────────────────────────────────

    async def detect_signature(self, page: np.ndarray) -> tuple[list[float], float] | None:
        """Detect the highest-confidence signature on a 640x640 page tensor.

        Returns (bbox, confidence) where bbox is [x1, y1, x2, y2] **normalised
        to [0, 1]**, or None if nothing clears the confidence floor.

        Normalised coordinates let the caller crop from the original
        full-resolution page rather than from the 640x640 detector input, which
        is what keeps the crop sharp enough for the extractors.
        """
        result = await self._infer(settings.yolo_model, "images", page, ["output0"])

        raw = result["output0"]
        if raw.ndim == 3:
            raw = raw[0]  # [1, 5, N] -> [5, N]
        if raw.size == 0 or raw.shape[1] == 0:
            return None

        confidences = raw[4, :]
        best = int(np.argmax(confidences))
        conf = float(confidences[best])
        if conf < settings.detection_confidence:
            logger.info("Best detection %.3f below floor %.3f", conf, settings.detection_confidence)
            return None

        cx, cy, w, h = (float(raw[i, best]) for i in range(4))
        x1 = max(0.0, (cx - w / 2) / YOLO_SIZE)
        y1 = max(0.0, (cy - h / 2) / YOLO_SIZE)
        x2 = min(1.0, (cx + w / 2) / YOLO_SIZE)
        y2 = min(1.0, (cy + h / 2) / YOLO_SIZE)
        if x2 <= x1 or y2 <= y1:
            logger.info("Degenerate bbox [%.3f %.3f %.3f %.3f]", x1, y1, x2, y2)
            return None

        return [x1, y1, x2, y2], conf

    # ── latest_net_G_B (CycleGAN denoiser) ────────────────────────────────────

    async def denoise(self, crop: np.ndarray) -> np.ndarray:
        """[1, 3, 224, 224] in [0, 1] -> denoised [1, 3, 224, 224] in [0, 1]."""
        result = await self._infer(settings.denoiser_model, "input", to_cyclegan(crop), ["output"])
        return from_cyclegan(result["output"])

    # ── resnet50_extractor + vgg16_extractor ──────────────────────────────────

    async def extract_features(self, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[1, 3, 224, 224] in [0, 1] -> two L2-normalised 4096-d embeddings."""
        caffe = to_caffe(clean)
        resnet_res, vgg_res = await asyncio.gather(
            self._infer(settings.resnet_model, "input_layer_1", caffe, ["fc1"]),
            self._infer(settings.vgg_model, "input_layer", caffe, ["fc1"]),
        )
        return (
            l2_normalise(resnet_res["fc1"].flatten()),
            l2_normalise(vgg_res["fc1"].flatten()),
        )


@lru_cache(maxsize=1)
def get_triton_service() -> TritonService:
    return TritonService()
