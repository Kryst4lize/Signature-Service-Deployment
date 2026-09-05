"""Generate config.pbtxt for the ONNX models this pipeline produces.

Shapes and tensor names are read back out of the exported ONNX graph rather
than hardcoded, so a config can never disagree with the model it sits next to —
which is the failure mode that produces a Triton startup error nobody can
explain.

Note on `max_batch_size` and `dims`: when max_batch_size >= 1, Triton treats the
batch dimension as implicit, so `dims` lists only the per-sample shape.
[3, 224, 224] therefore describes a [1, 3, 224, 224] request.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ONNX_TYPE_TO_TRITON = {
    1: "TYPE_FP32",
    2: "TYPE_UINT8",
    3: "TYPE_INT8",
    6: "TYPE_INT32",
    7: "TYPE_INT64",
    9: "TYPE_BOOL",
    10: "TYPE_FP16",
    11: "TYPE_FP64",
}


def _tensor_specs(tensors) -> list[tuple[str, str, list[int]]]:
    specs = []
    for tensor in tensors:
        dtype = _ONNX_TYPE_TO_TRITON.get(tensor.type.tensor_type.elem_type, "TYPE_FP32")
        # Skip the batch dimension; Triton supplies it implicitly.
        dims = [
            dim.dim_value if dim.dim_value > 0 else -1
            for dim in list(tensor.type.tensor_type.shape.dim)[1:]
        ]
        specs.append((tensor.name, dtype, dims))
    return specs


def _block(kind: str, name: str, dtype: str, dims: list[int]) -> str:
    return (
        f"{kind} {{\n"
        f'  name: "{name}"\n'
        f"  data_type: {dtype}\n"
        f"  dims: [ {', '.join(str(d) for d in dims)} ]\n"
        f"}}\n"
    )


def generate(
    onnx_path: str | Path,
    model_name: str,
    max_batch_size: int = 1,
    instance_kind: str = "KIND_GPU",
) -> str:
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    initialisers = {i.name for i in graph.initializer}
    real_inputs = [t for t in graph.input if t.name not in initialisers]

    parts = [
        f'name: "{model_name}"\n',
        'backend: "onnxruntime"\n',
        f"max_batch_size: {max_batch_size}\n\n",
    ]
    for name, dtype, dims in _tensor_specs(real_inputs):
        parts.append(_block("input", name, dtype, dims))
        parts.append("\n")
    for name, dtype, dims in _tensor_specs(graph.output):
        parts.append(_block("output", name, dtype, dims))
        parts.append("\n")

    # No dynamic_batching block. With max_batch_size 1 and
    # preferred_batch_size [1] it is inert — Triton dispatches immediately once
    # the pending batch matches a preferred size — so it only ever read as
    # configuration that was doing something.
    gpus = "\n    gpus: [ 0 ]" if instance_kind == "KIND_GPU" else ""
    parts.append(
        f"instance_group [\n  {{\n    kind: {instance_kind}{gpus}\n    count: 1\n  }}\n]\n"
    )
    return "".join(parts)


def write(
    onnx_path: str | Path,
    model_dir: str | Path,
    model_name: str,
    max_batch_size: int = 1,
    instance_kind: str = "KIND_GPU",
) -> Path:
    """Write `<model_dir>/config.pbtxt` describing `onnx_path`."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    config = generate(onnx_path, model_name, max_batch_size, instance_kind)
    target = model_dir / "config.pbtxt"
    target.write_text(config)
    logger.info("Wrote %s", target)
    return target
