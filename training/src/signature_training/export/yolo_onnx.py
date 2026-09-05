"""YOLOv8 detector -> ONNX.

This step was documented but had no implementation: the docs described
converting `yolov8s.pt` with `ultralytics.YOLO.export()`, while every converter
in the repo handled only Keras and CycleGAN. The `yolov8s/1/model.onnx` that
production runs had no reproducible provenance in the tree.

The output layout matters to the client. Ultralytics exports a single `output0`
of shape [1, 4 + nc, N] — rows 0-3 are cx, cy, w, h in *input pixel* space
(0..imgsz), and the remaining rows are per-class scores. With one class
(`signature`) that is [1, 5, N], which is what
inference/triton/model_repository/yolov8s/config.pbtxt declares and what
`TritonService.detect_signature` decodes.

NMS is deliberately not baked in: `detect_signature` takes the single
highest-confidence box, so the extra graph would be dead weight.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def export(
    weights: str | Path,
    onnx_path: str | Path,
    image_size: int = 640,
    opset: int = 13,
    simplify_graph: bool = True,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for the YOLO export step: pip install ultralytics"
        ) from exc

    weights, onnx_path = Path(weights), Path(onnx_path)
    if not weights.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}. Point paths.models at the "
            "directory holding your trained yolov8s.pt."
        )

    logger.info("Exporting %s at %dx%d", weights.name, image_size, image_size)
    model = YOLO(str(weights))
    produced = Path(
        model.export(
            format="onnx",
            imgsz=image_size,
            opset=opset,
            simplify=simplify_graph,
            # Dynamic batch dim, for the same reason as the Keras export: with
            # `max_batch_size: 1` Triton requires dim 0 to be dynamic. The
            # spatial dims stay fixed at imgsz; only the batch axis varies.
            dynamic=True,
            nms=False,  # the client picks the top box itself
        )
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != onnx_path.resolve():
        shutil.move(str(produced), onnx_path)

    _check_output_shape(onnx_path)
    return onnx_path


def _check_output_shape(onnx_path: Path) -> None:
    """Warn early if the class count does not match what the service decodes."""
    import onnx

    graph = onnx.load(onnx_path).graph
    for tensor in graph.output:
        dims = [d.dim_value or d.dim_param for d in tensor.type.tensor_type.shape.dim]
        logger.info("  output %s %s", tensor.name, dims)
        if tensor.name == "output0" and len(dims) == 3 and isinstance(dims[1], int):
            classes = dims[1] - 4
            if classes != 1:
                logger.warning(
                    "output0 has %d classes; the service assumes a single "
                    "'signature' class and reads row 4 as its confidence. "
                    "Update TritonService.detect_signature and the config.pbtxt "
                    "dims before deploying this model.",
                    classes,
                )
