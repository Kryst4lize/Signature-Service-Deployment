"""Keras feature extractors -> ONNX, in the layout Triton is configured for.

`inputs_as_nchw` is the whole point of this module. The Keras graph is NHWC;
every config.pbtxt declares `dims: [3, 224, 224]`, i.e. NCHW. The flag makes
tf2onnx insert a transpose at the graph input so Triton can send NCHW.

The repo previously had a second Keras->ONNX converter
(convert_model/convert_to_onnx_gemini.py) that omitted the flag and therefore
emitted an NHWC input — `['unk', 224, 224, 3]` instead of `[1, 3, 224, 224]` —
which Triton rejects against these configs. That converter is deleted; this is
the only Keras export path.

No preprocessing is baked into the graph. The extractors were trained with
Keras `preprocess_input(mode="caffe")` applied *outside* the model by
ImageDataGenerator, so the ONNX input is the already-preprocessed tensor and the
serving side must reproduce it. See inference/api/app/triton.py:to_caffe.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Input tensor names, which must match the `input` block of each config.pbtxt.
# They are the names Keras assigned when the models were first built; renaming
# one means regenerating the corresponding config.
INPUT_NAMES = {
    "vgg16": "input_layer",
    "resnet50": "input_layer_1",
}
OUTPUT_NAME = "fc1"


def convert(
    keras_path: str | Path,
    onnx_path: str | Path,
    input_name: str,
    image_size: int = 224,
    opset: int = 13,
    simplify_graph: bool = True,
) -> Path:
    import tensorflow as tf
    import tf2onnx
    import onnx

    keras_path, onnx_path = Path(keras_path), Path(onnx_path)
    if not keras_path.is_file():
        raise FileNotFoundError(f"Keras model not found: {keras_path}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting %s -> %s", keras_path.name, onnx_path.name)
    model = tf.keras.models.load_model(keras_path, compile=False)

    # Batch dimension must be None, not 1. Triton's ONNX Runtime backend
    # validates shapes at load time: with `max_batch_size: 1` in config.pbtxt it
    # treats dim 0 as the implicit batch dim and requires the model to declare
    # it dynamic. A model exported with a literal 1 is rejected —
    #   "model configuration specified max-batch 1 but the model does not
    #    support batching"
    # — so the extractors would fail to load even though every other field
    # matched.
    spec = (tf.TensorSpec((None, image_size, image_size, 3), tf.float32, name=input_name),)
    proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset,
        inputs_as_nchw=[input_name],
    )

    if simplify_graph:
        proto = _simplify(proto)

    onnx.save(proto, onnx_path)
    _log_io(onnx_path)
    return onnx_path


def _simplify(proto):
    """Fold constants and drop no-op nodes.

    Uses onnxslim, which is what requirements.txt has always pinned. The old
    script imported `onnxsim` instead — a different project — so a clean install
    died with ModuleNotFoundError before converting anything. onnxslim is the
    right side of that mismatch to keep: it is pure Python with wheels for every
    platform, whereas onnxsim has no aarch64 wheel and needs cmake and a C++
    toolchain to build from source.

    Still guarded, so simplification degrades to a warning rather than failing
    an otherwise complete export.
    """
    try:
        import onnxslim
    except ImportError:
        logger.warning("onnxslim not installed - skipping graph simplification")
        return proto

    try:
        simplified = onnxslim.slim(proto)
    except Exception as exc:  # noqa: BLE001 - simplification is best-effort
        logger.warning("Simplification failed (%s) - keeping the original graph", exc)
        return proto

    logger.info("Graph simplified")
    return simplified


def _log_io(onnx_path: Path) -> None:
    import onnx

    model = onnx.load(onnx_path)
    for tensor in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in tensor.type.tensor_type.shape.dim]
        logger.info("  input  %s %s", tensor.name, dims)
    for tensor in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in tensor.type.tensor_type.shape.dim]
        logger.info("  output %s %s", tensor.name, dims)


def convert_extractors(
    models_dir: Path, onnx_dir: Path, cfg, backbones: list[str] | None = None
) -> dict[str, Path]:
    """Convert whichever of the requested extractors exist in `models_dir`."""
    produced = {}
    for backbone in (backbones if backbones is not None else list(INPUT_NAMES)):
        input_name = INPUT_NAMES[backbone]
        src = models_dir / f"{backbone}_extractor.keras"
        if not src.is_file():
            logger.info("Skipping %s - %s not found", backbone, src.name)
            continue
        produced[f"{backbone}_extractor"] = convert(
            src,
            onnx_dir / f"{backbone}_extractor.onnx",
            input_name=input_name,
            image_size=cfg.export.image_size,
            opset=cfg.export.opset,
            simplify_graph=cfg.export.simplify,
        )
    return produced
