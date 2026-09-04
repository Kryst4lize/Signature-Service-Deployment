#!/usr/bin/env python3
"""
convert_to_trt.py
=================
Modular converter: .keras / .onnx / .pth / .pt  →  TensorRT .engine
+ Triton Inference Server config.pbtxt generator.

Usage
-----
# You know the shape:
python convert_to_trt.py \
    --model      path/to/model.onnx \
    --output-dir ./triton_repo \
    --input-shapes "images:3x640x640" \
    --max-batch  8 \
    --precision  fp16

# You only know it's an image model (shape auto-detected):
python convert_to_trt.py \
    --model      path/to/model.onnx \
    --output-dir ./triton_repo \
    --image-input \
    --precision  fp16

Shape resolution order (when --input-shapes is omitted / --image-input used)
-----------------------------------------------------------------------------
  1. Read from the model file itself (ONNX graph, Keras .input_shape, torchinfo)
  2. If dynamic/unknown: probe common image resolutions (224, 256, 384, 512, 640)
     → prints a menu and asks the user to confirm
  3. Falls back to 3×224×224 with a loud warning if running non-interactively

Supported source formats
------------------------
  .onnx        → direct TRT conversion
  .keras / .h5 → Keras → ONNX (tf2onnx) → TRT
  .pt  / .pth  → PyTorch → ONNX (torch.onnx) → TRT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("convert_to_trt")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Common image resolutions to probe when the shape cannot be read from the model.
# Format: (channels, height, width)  — CHW, the TRT/PyTorch convention.
COMMON_IMAGE_SHAPES: List[Tuple[int, int, int]] = [
    (3, 224, 224),   # ImageNet standard
    (3, 256, 256),
    (3, 320, 320),
    (3, 384, 384),
    (3, 416, 416),   # YOLO classic
    (3, 512, 512),
    (3, 608, 608),
    (3, 640, 640),   # YOLOv5/v8 default
    (3, 768, 768),
    (3, 1024, 1024),
    (1, 224, 224),   # grayscale variants
    (1, 256, 256),
    (1, 512, 512),
]

_ONNX_DTYPE_TO_TRITON = {
    1:  "TYPE_FP32",
    2:  "TYPE_UINT8",
    3:  "TYPE_INT8",
    5:  "TYPE_INT32",
    6:  "TYPE_INT32",
    7:  "TYPE_INT64",
    9:  "TYPE_BOOL",
    10: "TYPE_FP16",
    11: "TYPE_FP64",
    12: "TYPE_UINT32",
}

_NP_DTYPE_TO_TRITON = {
    "float32": "TYPE_FP32",
    "float16": "TYPE_FP16",
    "int8":    "TYPE_INT8",
    "int32":   "TYPE_INT32",
    "int64":   "TYPE_INT64",
    "uint8":   "TYPE_UINT8",
    "bool":    "TYPE_BOOL",
}


# ──────────────────────────────────────────────────────────────────────────────
# Shape detector
# ──────────────────────────────────────────────────────────────────────────────

class ShapeDetector:
    """
    Attempts to read input tensor shapes directly from the model file.
    Falls back to an interactive menu of common image resolutions.

    Call:
        specs = ShapeDetector(model_path).detect(input_name="input")
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.suffix = model_path.suffix.lower()

    # ── public entry point ─────────────────────────────────────────────────
    def detect(
        self,
        input_name: Optional[str] = None,
        *,
        user_supplied_name: bool = False,
    ) -> List["TensorSpec"]:
        """
        Returns a list of TensorSpec (one per model input).

        Parameters
        ----------
        input_name : str | None
            Name to use when the model has no embedded tensor names.
            Only applied when ``user_supplied_name=True``; otherwise the
            name read from the model file is preserved as-is.
        user_supplied_name : bool
            Set True only when the user explicitly typed a name via
            --image-input <name>.  Default False preserves model graph names.
        """
        specs: Optional[List[TensorSpec]] = None

        if self.suffix == ".onnx":
            specs = self._from_onnx()
        elif self.suffix in (".keras", ".h5"):
            specs = self._from_keras()
        elif self.suffix in (".pt", ".pth"):
            specs = self._from_pytorch()

        if specs:
            # Only stomp over the model's own tensor name when the user
            # explicitly asked us to (--image-input myname).
            if user_supplied_name and input_name and len(specs) == 1:
                specs[0].name = input_name
            log.info(f"Auto-detected input specs: {[(s.name, s.dims) for s in specs]}")
            return specs

        # ── Could not read from model → ask user ──────────────────────────
        log.warning(
            "Could not read input shapes from the model file. "
            "Falling back to image-shape probe."
        )
        return self._prompt_image_shape(input_name or "input")

    # ── ONNX ──────────────────────────────────────────────────────────────
    def _from_onnx(self) -> Optional[List["TensorSpec"]]:
        try:
            import onnx  # type: ignore
            model = onnx.load(str(self.model_path))
            specs = []
            for inp in model.graph.input:
                t = inp.type.tensor_type
                dims = []
                all_static = True
                for i, d in enumerate(t.shape.dim):
                    if i == 0:
                        continue  # skip batch
                    val = d.dim_value
                    if val <= 0:          # dynamic dim (symbolic or 0)
                        all_static = False
                        dims.append(-1)
                    else:
                        dims.append(val)
                dtype = _ONNX_DTYPE_TO_TRITON.get(t.elem_type, "TYPE_FP32")
                specs.append(TensorSpec(name=inp.name, dims=dims, dtype=dtype))

            if not specs:
                return None

            # If any dim is dynamic and looks like an image (3 or 4 dims excl. batch)
            has_dynamic = any(-1 in s.dims for s in specs)
            if has_dynamic:
                log.warning(
                    "ONNX graph has dynamic input dimensions: "
                    + str([(s.name, s.dims) for s in specs])
                )
                # Return what we have; pipeline will call _resolve_dynamic
                return specs
            return specs
        except Exception as e:
            log.debug(f"ONNX shape detection failed: {e}")
            return None

    # ── Keras ─────────────────────────────────────────────────────────────
    def _from_keras(self) -> Optional[List["TensorSpec"]]:
        # Suppress verbose TF GPU / cuDNN warnings that flood the terminal
        # and can mask the real shape-detection log messages.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        try:
            import tensorflow as tf  # type: ignore
            model = tf.keras.models.load_model(str(self.model_path), compile=False)
            specs = []
            inputs = (
                model.inputs if hasattr(model, "inputs") else [model.input]
            )
            for inp in inputs:
                raw_shape = inp.shape.as_list() if hasattr(inp.shape, "as_list") else list(inp.shape) # e.g. [None, 224, 224, 3]
                spatial = raw_shape[1:]           # drop batch dim

                if len(spatial) == 3:
                    # NHWC (H, W, C) → CHW (C, H, W) for TRT/pbtxt
                    dims = self._nhwc_to_chw(spatial)
                    log.info(
                        f"Keras input '{inp.name}' raw NHWC shape "
                        f"{raw_shape} → stored as CHW {dims}"
                    )
                else:
                    dims = [-1 if (d is None or d <= 0) else d for d in spatial]

                dtype = _NP_DTYPE_TO_TRITON.get(
                    inp.dtype.name if hasattr(inp.dtype, "name") else str(inp.dtype),
                    "TYPE_FP32",
                )
                name = inp.name.split(":")[0]
                specs.append(TensorSpec(name=name, dims=dims, dtype=dtype))

            if specs:
                return specs

            log.warning("Keras model loaded but no input tensors found.")
            return None

        except Exception as e:
            # Log at WARNING so the user sees WHY auto-detection fell back to
            # the menu (previously logged at DEBUG and silently disappeared).
            log.warning(f"Keras shape auto-detection failed: {e}")
            log.warning(
                "This is often caused by missing GPU libraries (cuDNN). "
                "Shape detection still works on CPU — if the error above is "
                "a cuDNN warning rather than a real exception, try setting "
                "TF_CPP_MIN_LOG_LEVEL=3 and re-running."
            )
            return None

    # ── PyTorch ───────────────────────────────────────────────────────────
    def _from_pytorch(self) -> Optional[List["TensorSpec"]]:
        """
        Uses torchinfo (if installed) to inspect the first layer for input shape.
        Falls back to reading a 'input_shape' key sometimes stored in checkpoints.
        """
        # ── Ultralytics / YOLO fast-path ──────────────────────────────────
        try:
            if _is_ultralytics_checkpoint(self.model_path):
                import torch
                ckpt = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
                # Try reading imgsz from training args or model args
                imgsz = None
                for container in (ckpt.get("train_args"), ckpt.get("args")):
                    if isinstance(container, dict):
                        imgsz = container.get("imgsz")
                    elif hasattr(container, "imgsz"):
                        imgsz = container.imgsz
                    if imgsz: break
                if isinstance(imgsz, (list, tuple)):
                    h, w = int(imgsz[0]), int(imgsz[-1])
                elif imgsz:
                    h = w = int(imgsz)
                else:
                    h = w = 640  # YOLOv8 default
                log.info(f"Ultralytics checkpoint detected: imgsz={h}×{w}")
                return [TensorSpec(name="images", dims=[3, h, w])]
        except Exception as e:
            log.debug(f"Ultralytics shape probe failed: {e}")

        try:
            import torch  # type: ignore
            ckpt = torch.load(str(self.model_path), map_location="cpu", weights_only=False)

            # Some training scripts stash the input shape in the checkpoint dict
            for key in ("input_shape", "input_size", "img_size"):
                if isinstance(ckpt, dict) and key in ckpt:
                    val = ckpt[key]
                    # normalise to list of ints
                    if isinstance(val, int):
                        dims = [3, val, val]
                    elif len(val) == 2:
                        dims = [3, int(val[0]), int(val[1])]
                    else:
                        dims = [int(d) for d in val]
                    log.info(f"Found '{key}' in checkpoint: {dims}")
                    return [TensorSpec(name="input", dims=dims)]
        except Exception as e:
            log.debug(f"PyTorch checkpoint probe failed: {e}")

        # torchinfo path (best-effort; may not work for all architectures)
        try:
            import torch
            import torchinfo  # type: ignore
            ckpt = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
            model = ckpt if isinstance(ckpt, torch.nn.Module) else ckpt.get("model")
            if model is None:
                return None
            model.eval()
            # Try the most common resolution first; torchinfo will tell us if it works
            for shape in COMMON_IMAGE_SHAPES:
                try:
                    info = torchinfo.summary(model, input_size=(1, *shape), verbose=0)
                    log.info(f"torchinfo probed shape {shape} successfully.")
                    return [TensorSpec(name="input", dims=list(shape))]
                except Exception:
                    continue
        except ImportError:
            log.debug("torchinfo not installed; skipping PyTorch shape probe.")
        except Exception as e:
            log.debug(f"torchinfo probe failed: {e}")

        return None

    # ── Interactive fallback ───────────────────────────────────────────────
    def _prompt_image_shape(self, input_name: str) -> List["TensorSpec"]:
        """
        Prints a numbered menu of COMMON_IMAGE_SHAPES and waits for the user
        to pick one.  If stdin is not a TTY (CI / Docker), uses 3×224×224
        and emits a loud warning.
        """
        if not sys.stdin.isatty():
            default = (3, 224, 224)
            log.warning(
                f"Non-interactive mode: defaulting to shape {default}. "
                "Re-run with --input-shapes to override."
            )
            return [TensorSpec(name=input_name, dims=list(default))]

        print("\n" + "─" * 60)
        print("Could not detect input shape automatically.")
        print("Select the image shape that matches your model:\n")
        for i, (c, h, w) in enumerate(COMMON_IMAGE_SHAPES, 1):
            label = f"{c}×{h}×{w}"
            extra = ""
            if (c, h, w) == (3, 224, 224):
                extra = "  ← ImageNet standard"
            elif (c, h, w) == (3, 640, 640):
                extra = "  ← YOLOv5/v8 default"
            elif (c, h, w) == (3, 416, 416):
                extra = "  ← YOLOv4 / Darknet"
            print(f"  [{i:2d}] {label}{extra}")
        print(f"  [ c] custom …")
        print("─" * 60)

        while True:
            raw = input("Enter choice: ").strip().lower()
            if raw == "c":
                return [self._prompt_custom_shape(input_name)]
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(COMMON_IMAGE_SHAPES):
                    chosen = list(COMMON_IMAGE_SHAPES[idx])
                    print(f"  → Using shape {chosen}")
                    return [TensorSpec(name=input_name, dims=chosen)]
            except ValueError:
                pass
            print("  Invalid choice. Try again.")

    @staticmethod
    def _prompt_custom_shape(input_name: str) -> "TensorSpec":
        """Ask user to type C H W manually."""
        while True:
            raw = input("Enter shape as C H W (e.g. 3 640 640): ").strip()
            try:
                parts = [int(x) for x in raw.split()]
                if len(parts) == 3:
                    return TensorSpec(name=input_name, dims=parts)
                if len(parts) == 2:
                    return TensorSpec(name=input_name, dims=[3] + parts)
            except ValueError:
                pass
            print("  Could not parse. Enter 2 or 3 integers.")

    # ── Utility ───────────────────────────────────────────────────────────
    @staticmethod
    def _nhwc_to_chw(dims: List[Optional[int]]) -> List[int]:
        """
        Convert Keras NHWC dims (H, W, C) → CHW (C, H, W).
        Replaces None with -1 for dynamic axes.
        """
        clean = [-1 if (d is None or d <= 0) else d for d in dims]
        if len(clean) == 3:                # H W C → C H W
            return [clean[2], clean[0], clean[1]]
        return clean                       # leave other shapes as-is


def _pick_default_spatial(c: int) -> Tuple[int, int]:
    """Return a sensible (H, W) default for a known channel count."""
    return (640, 640) if c != 1 else (256, 256)


def _resolve_dynamic_dims(
    specs: List["TensorSpec"],
    detector: ShapeDetector,
) -> List["TensorSpec"]:
    """
    Replace -1 dims with concrete values.

    Pattern          Action
    ───────────────────────────────────────────────────────────────────────
    [C, -1, -1]      C known, spatial dynamic → default 640×640 (or 256 for
                     grayscale).  Logs INFO + tells user how to override.
    [-1, -1, -1]     Everything dynamic       → interactive menu / fallback.
    other + -1       Non-image               → warn, keep as-is.
    """
    resolved = []
    for s in specs:
        if -1 not in s.dims:
            resolved.append(s)
            continue

        if len(s.dims) == 3:
            c, h, w = s.dims

            # Only H/W are dynamic — C is known
            if c > 0 and h == -1 and w == -1:
                dh, dw = _pick_default_spatial(c)
                log.info(
                    f"Input '{s.name}' has dynamic spatial dims [{c}, -1, -1]. "
                    f"Defaulting to [{c}, {dh}, {dw}]. "
                    f"Override with --input-shapes \"{s.name}:{c}x<H>x<W>\"."
                )
                resolved.append(TensorSpec(name=s.name, dims=[c, dh, dw], dtype=s.dtype))
                continue

            # C is also dynamic — need the full interactive menu
            log.warning(
                f"Input '{s.name}' has fully dynamic dims {s.dims}. "
                "Entering image-shape selection …"
            )
            replacement = detector._prompt_image_shape(s.name)[0]
            replacement.dtype = s.dtype
            resolved.append(replacement)
        else:
            log.warning(
                f"Input '{s.name}' has dynamic dims {s.dims}. "
                "You may need to specify --input-shapes explicitly."
            )
            resolved.append(s)
    return resolved


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TensorSpec:
    """Describes a single input or output tensor."""
    name: str
    dims: List[int]          # static dims (batch dim excluded)
    dtype: str = "TYPE_FP32" # Triton TYPE_* constant

    # Convenience ----------------------------------------------------------
    @classmethod
    def from_str(cls, spec: str) -> "TensorSpec":
        """Parse 'name:d0xd1x…' → TensorSpec.  Batch dim is NOT included."""
        name, shape_str = spec.split(":", 1)
        dims = [int(d) for d in shape_str.split("x")]
        return cls(name=name, dims=dims)

    def triton_dims(self) -> List[int]:
        """Dims as Triton expects them (no explicit batch axis)."""
        return self.dims


@dataclass
class ConversionConfig:
    """Everything the pipeline needs to know."""
    source_path: Path
    output_dir: Path
    model_name: str
    input_specs: List[TensorSpec]
    output_specs: List[TensorSpec] = field(default_factory=list)
    max_batch_size: int = 1
    precision: str = "fp32"          # fp32 | fp16 | int8
    workspace_mb: int = 4096
    device: int = 0
    triton_version: int = 1          # model version folder inside output_dir
    image_input_name: Optional[str] = None  # hint used by ShapeDetector
    arch_name:  Optional[str] = None  # key into ARCH_REGISTRY
    model_module: Optional[str] = None  # 'path/to/file.py::ClassName'

    # Derived ---------------------------------------------------------------
    @property
    def onnx_path(self) -> Path:
        return self.output_dir / "model.onnx"

    @property
    def engine_path(self) -> Path:
        version_dir = self.output_dir / str(self.triton_version)
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir / "model.plan"

    @property
    def pbtxt_path(self) -> Path:
        return self.output_dir / "config.pbtxt"


# ──────────────────────────────────────────────────────────────────────────────
# Base converter
# ──────────────────────────────────────────────────────────────────────────────
class BaseConverter(ABC):
    """Abstract base – subclass for each source format."""

    def __init__(self, cfg: ConversionConfig):
        self.cfg = cfg

    @abstractmethod
    def to_onnx(self, tmp_dir: Path) -> Path:
        """Return path to the exported .onnx file."""

    # ── shared helper ──────────────────────────────────────────────────────
    def _check_import(self, module: str, install_hint: str) -> None:
        import importlib
        if importlib.util.find_spec(module) is None:
            raise ImportError(
                f"Module '{module}' not found. Install with: {install_hint}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Format-specific converters
# ──────────────────────────────────────────────────────────────────────────────
class OnnxConverter(BaseConverter):
    """Source is already .onnx – nothing to do."""

    def to_onnx(self, tmp_dir: Path) -> Path:
        log.info("Source is already ONNX – skipping export step.")
        return self.cfg.source_path


class KerasConverter(BaseConverter):
    """
    Keras .keras / .h5  →  ONNX via tf2onnx.

    Shape convention
    ────────────────
    TensorSpec.dims are stored in CHW order (C, H, W) — the canonical format
    used by TensorRT and the pbtxt config.  Keras itself expects NHWC tensors,
    so we convert CHW → HWC when building the tf.TensorSpec input signature.

    Stored in cfg   : [C, H, W]  e.g. [3, 224, 224]
    Passed to Keras : [None, H, W, C]  e.g. [None, 224, 224, 3]
    Written to pbtxt: [C, H, W]  e.g. [3, 224, 224]
    """

    # Dtype mapping: TF dtype string → tf.dtypes object
    _TF_DTYPE_MAP = {
        "TYPE_FP32": "float32",
        "TYPE_FP16": "float16",
        "TYPE_INT8":  "int8",
        "TYPE_INT32": "int32",
        "TYPE_INT64": "int64",
        "TYPE_UINT8": "uint8",
        "TYPE_BOOL":  "bool",
    }

    def to_onnx(self, tmp_dir: Path) -> Path:
        self._check_import("tensorflow", "pip install tensorflow")
        self._check_import("tf2onnx", "pip install tf2onnx")

        import tensorflow as tf  # type: ignore
        import tf2onnx             # type: ignore

        log.info("Loading Keras model …")
        model = tf.keras.models.load_model(str(self.cfg.source_path))

        # ── Verify / auto-correct dims from the actual model ───────────────
        # If the user typed a CHW shape from the menu we trust it, but we
        # also double-check against model.inputs so we can give a clear error
        # instead of the cryptic "found shape=(None,3,224,224)" message.
        self._validate_specs_against_model(model)

        # ── Build tf.TensorSpec in NHWC for Keras ─────────────────────────
        tf_specs = []
        input_names = []
        for s in self.cfg.input_specs:
            nhwc_shape = self._chw_to_nhwc(s.dims)
            tf_dtype   = self._TF_DTYPE_MAP.get(s.dtype, "float32")
            tf_specs.append(
                tf.TensorSpec(shape=nhwc_shape, dtype=tf_dtype, name=s.name)
            )
            input_names.append(s.name)
            log.info(
                f"  Input '{s.name}': stored as CHW {s.dims}, "
                f"passing NHWC {nhwc_shape} to Keras"
            )

        onnx_path = tmp_dir / "model.onnx"
        log.info("Converting Keras → ONNX …")
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=tf_specs,
            opset=17,
            output_path=str(onnx_path),
            inputs_as_nchw=input_names,
        )
        log.info(f"ONNX saved to {onnx_path}")
        return onnx_path

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _chw_to_nhwc(chw_dims: List[int]) -> List[Optional[int]]:
        """
        [C, H, W] → [None, H, W, C]   (None = dynamic batch)
        Non-image shapes (len != 3) are returned as [None, *dims] unchanged.
        """
        if len(chw_dims) == 3:
            c, h, w = chw_dims
            return [None, h, w, c]
        return [None] + chw_dims

    def _validate_specs_against_model(self, model) -> None:
        """
        Compare cfg.input_specs (CHW) against the model's actual NHWC shapes.
        Logs a clear warning if they differ; auto-corrects when possible.
        """
        try:
            raw_inputs = model.inputs if hasattr(model, "inputs") else [model.input]
        except Exception:
            return  # can't inspect; skip validation

        for spec, layer_inp in zip(self.cfg.input_specs, raw_inputs):
            actual_nhwc = layer_inp.shape.as_list() if hasattr(layer_inp.shape, "as_list") else list(layer_inp.shape)  # e.g. [None, 224, 224, 3]
            if None in actual_nhwc or len(actual_nhwc) < 4:
                continue  # fully dynamic or non-image; trust the user

            _, h_act, w_act, c_act = actual_nhwc[:4]
            c_spec, h_spec, w_spec = (spec.dims + [None, None, None])[:3]

            if (c_spec, h_spec, w_spec) != (c_act, h_act, w_act):
                log.warning(
                    f"Input spec mismatch for '{spec.name}':\n"
                    f"  You provided  (CHW): {spec.dims}\n"
                    f"  Model expects (CHW): [{c_act}, {h_act}, {w_act}]\n"
                    f"  Auto-correcting to model shape."
                )
                spec.dims = [c_act, h_act, w_act]


# ──────────────────────────────────────────────────────────────────────────────
# Architecture registry  (state-dict-only checkpoints need a model class)
# ──────────────────────────────────────────────────────────────────────────────

def _make_cyclegan_resnet_generator(input_nc=3, output_nc=3, ngf=64, n_blocks=9):
    """CycleGAN ResnetGenerator (9-block default)."""
    import torch.nn as nn

    class ResnetBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
                nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
            )
        def forward(self, x): return x + self.block(x)

    layers = [
        nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
        nn.Conv2d(ngf,   ngf*2, 3, 2, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
        nn.Conv2d(ngf*2, ngf*4, 3, 2, 1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
    ]
    for _ in range(n_blocks):
        layers.append(ResnetBlock(ngf*4))
    layers += [
        nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
        nn.ConvTranspose2d(ngf*2, ngf,   3, 2, 1, 1), nn.InstanceNorm2d(ngf),   nn.ReLU(True),
        nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh(),
    ]
    return nn.Sequential(*layers)


def _make_cyclegan_discriminator(input_nc=3, ndf=64, n_layers=3):
    """CycleGAN NLayerDiscriminator (PatchGAN)."""
    import torch.nn as nn

    def _block(in_f, out_f, stride, norm=True):
        layers = [nn.Conv2d(in_f, out_f, 4, stride, 1, bias=not norm)]
        if norm: layers.append(nn.InstanceNorm2d(out_f))
        layers.append(nn.LeakyReLU(0.2, True))
        return layers

    layers = _block(input_nc, ndf, 2, norm=False)
    nf = ndf
    for _ in range(1, n_layers):
        nf_prev, nf = nf, min(nf * 2, 512)
        layers += _block(nf_prev, nf, 2)
    nf_prev, nf = nf, min(nf * 2, 512)
    layers += _block(nf_prev, nf, 1)
    layers.append(nn.Conv2d(nf, 1, 4, 1, 1))
    return nn.Sequential(*layers)


def _make_cyclegan_pixel_discriminator(input_nc=3, ndf=64):
    """CycleGAN PixelDiscriminator."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(input_nc, ndf, 1, 1, 0), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf, ndf*2, 1, 1, 0, bias=False), nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf*2, 1, 1, 1, 0),
    )


def _torchvision_factory(name: str):
    """Return a factory lambda for a torchvision model."""
    def _factory():
        import torchvision.models as M  # type: ignore
        ctor = getattr(M, name, None)
        if ctor is None:
            raise ValueError(f"torchvision has no model '{name}'.")
        return ctor(weights=None)
    return _factory


# Registry maps --arch value → zero-arg factory that returns an nn.Module
ARCH_REGISTRY: Dict[str, Any] = {
    # CycleGAN ----------------------------------------------------------------
    "cyclegan_resnet_9":   lambda: _make_cyclegan_resnet_generator(n_blocks=9),
    "cyclegan_resnet_6":   lambda: _make_cyclegan_resnet_generator(n_blocks=6),
    "cyclegan_discriminator":   _make_cyclegan_discriminator,
    "cyclegan_discriminator_1": lambda: _make_cyclegan_discriminator(n_layers=1),
    "cyclegan_pixel_disc":      _make_cyclegan_pixel_discriminator,
    # torchvision shortcuts ---------------------------------------------------
    "resnet18":   _torchvision_factory("resnet18"),
    "resnet34":   _torchvision_factory("resnet34"),
    "resnet50":   _torchvision_factory("resnet50"),
    "resnet101":  _torchvision_factory("resnet101"),
    "vgg16":      _torchvision_factory("vgg16"),
    "vgg19":      _torchvision_factory("vgg19"),
    "densenet121":_torchvision_factory("densenet121"),
    "mobilenet_v2":_torchvision_factory("mobilenet_v2"),
    "efficientnet_b0":_torchvision_factory("efficientnet_b0"),
    "efficientnet_b4":_torchvision_factory("efficientnet_b4"),
}


# Key-pattern heuristics: if these substrings appear in the state-dict keys
# we can guess the architecture automatically.
_KEY_HINTS: List[Tuple[str, str]] = [
    # (substring_in_key,         arch_registry_name)
    ("model.0.weight",           "cyclegan_resnet_9"),   # CycleGAN generator first layer
    ("model.3.weight",           "cyclegan_resnet_9"),
    ("model.0.bias",             "cyclegan_resnet_9"),
    ("model.1.weight",           "cyclegan_discriminator"),  # PatchGAN
    ("layer1.0.conv1.weight",    "resnet18"),
    ("layer1.0.conv1.weight",    "resnet50"),
    ("features.0.weight",        "vgg16"),
    ("features.0.weight",        "vgg19"),
    ("conv0.weight",             "densenet121"),
    ("features.0.0.weight",      "mobilenet_v2"),
]


def _guess_arch_from_state_dict(state_dict: dict) -> Optional[str]:
    """
    Heuristically guess the architecture name from state-dict key patterns.
    Returns an ARCH_REGISTRY key or None if nothing matches.
    """
    keys = list(state_dict.keys())
    key_set = set(keys)

    # CycleGAN ResnetGenerator: has keys like 'model.X.weight'
    # and ConvTranspose2d layers (numbered high)
    model_keys = [k for k in keys if k.startswith("model.") and k.endswith(".weight")]
    if model_keys:
        indices = []
        for k in model_keys:
            try: indices.append(int(k.split(".")[1]))
            except ValueError: pass
        if indices and max(indices) > 20:
            log.info("Heuristic: looks like CycleGAN ResnetGenerator (9-block).")
            return "cyclegan_resnet_9"
        if indices and max(indices) > 14:
            log.info("Heuristic: looks like CycleGAN ResnetGenerator (6-block).")
            return "cyclegan_resnet_6"
        # Small index range → likely discriminator
        if indices and max(indices) <= 14:
            log.info("Heuristic: looks like CycleGAN NLayerDiscriminator.")
            return "cyclegan_discriminator"

    if "layer1.0.conv1.weight" in key_set:
        # ResNet — distinguish by depth from total key count
        n = len(keys)
        if n < 80:   return "resnet18"
        if n < 120:  return "resnet34"
        if n < 170:  return "resnet50"
        return "resnet101"

    if any(k.startswith("features.0.weight") for k in keys):
        return "vgg16"

    if "conv0.weight" in key_set:
        return "densenet121"

    if "features.0.0.weight" in key_set:
        return "mobilenet_v2"

    return None


def _load_model_module(spec: str) -> type:
    """
    Import a class from an external file.
    Spec format:  "path/to/module.py::ClassName"
    """
    if "::" not in spec:
        raise ValueError(
            f"--model-module must be in the form 'path/to/file.py::ClassName', got: {spec!r}"
        )
    module_path, class_name = spec.rsplit("::", 1)
    module_path = Path(module_path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")

    import importlib.util
    mod_spec = importlib.util.spec_from_file_location("_user_module", module_path)
    module   = importlib.util.module_from_spec(mod_spec)  # type: ignore[arg-type]
    mod_spec.loader.exec_module(module)                   # type: ignore[union-attr]

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Class '{class_name}' not found in {module_path}. "
            f"Available names: {[n for n in dir(module) if not n.startswith('_')]}"
        )
    log.info(f"Loaded custom class '{class_name}' from {module_path}")
    return cls


class PytorchConverter(BaseConverter):
    """
    PyTorch .pt / .pth  →  ONNX via torch.onnx.

    Architecture resolution order
    ─────────────────────────────
    1. Checkpoint already IS an nn.Module (full model saved with torch.save(model, …))
    2. Checkpoint dict has a 'model' key containing an nn.Module
    3. cfg.model_module  →  load class from external .py file  (file.py::ClassName)
    4. cfg.arch_name     →  look up ARCH_REGISTRY
    5. Auto-guess from state-dict key patterns (_guess_arch_from_state_dict)
    6. Raise with a helpful error listing available --arch names
    """

    def to_onnx(self, tmp_dir: Path) -> Path:
        self._check_import("torch", "pip install torch")
        import torch  # type: ignore

        log.info("Loading PyTorch checkpoint …")
        checkpoint = torch.load(
            str(self.cfg.source_path),
            map_location="cpu",
            weights_only=False,
        )

        model = self._resolve_model(checkpoint, torch)
        model.eval()

        # Build dummy inputs
        dummy_inputs = []
        for s in self.cfg.input_specs:
            shape = [self.cfg.max_batch_size] + s.dims
            dummy_inputs.append(torch.zeros(*shape))

        onnx_path   = tmp_dir / "model.onnx"
        input_names = [s.name for s in self.cfg.input_specs]
        output_names = (
            [s.name for s in self.cfg.output_specs]
            if self.cfg.output_specs else ["output"]
        )
        dynamic_axes = {n: {0: "batch"} for n in input_names + output_names}

        log.info("Exporting PyTorch → ONNX …")
        torch.onnx.export(
            model,
            tuple(dummy_inputs) if len(dummy_inputs) > 1 else dummy_inputs[0],
            str(onnx_path),
            opset_version=17,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        log.info(f"ONNX saved to {onnx_path}")
        return onnx_path

    # ── architecture resolution ────────────────────────────────────────────
    def _resolve_model(self, checkpoint, torch) -> "torch.nn.Module":
        cfg = self.cfg

        # 1. Full model saved directly
        if isinstance(checkpoint, torch.nn.Module):
            log.info("Checkpoint is a full nn.Module — no architecture lookup needed.")
            return checkpoint

        # 2. Dict with 'model' key holding an nn.Module
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), torch.nn.Module):
            log.info("Found nn.Module under checkpoint['model'].")
            return checkpoint["model"]

        # Normalise state dict
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or (checkpoint if isinstance(checkpoint, dict) else None)
        )
        if state_dict is None or not isinstance(state_dict, dict):
            raise ValueError(
                "Cannot extract a state_dict from the checkpoint. "
                "Save with torch.save(model.state_dict(), path) or torch.save(model, path)."
            )

        # 3. External module file
        if cfg.model_module:
            cls = _load_model_module(cfg.model_module)
            return self._load_state(cls(), state_dict)

        # 4. Explicit arch name
        if cfg.arch_name:
            return self._from_registry(cfg.arch_name, state_dict)

        # 5. Auto-guess from key patterns
        guessed = _guess_arch_from_state_dict(state_dict)
        if guessed:
            log.info(f"Auto-detected architecture: '{guessed}'")
            return self._from_registry(guessed, state_dict)

        # 6. Give up with a helpful error
        available = "\n    ".join(sorted(ARCH_REGISTRY))
        raise ValueError(
            "\n"
            "Could not determine the model architecture from the checkpoint.\n"
            "The checkpoint appears to be a state-dict only (no nn.Module embedded).\n\n"
            "Fix options:\n"
            "  A) Pass --arch <name>  — choose from the built-in registry:\n"
            f"    {available}\n\n"
            "  B) Pass --model-module path/to/your_model.py::YourClassName\n"
            "     (the class must accept no required constructor arguments, or\n"
            "      you can hard-code defaults in your file)\n\n"
            "  C) Re-save the model with the architecture embedded:\n"
            "     torch.save(model, 'model_full.pth')  # instead of state_dict"
        )

    def _from_registry(self, name: str, state_dict: dict) -> "torch.nn.Module":
        factory = ARCH_REGISTRY.get(name)
        if factory is None:
            available = ", ".join(sorted(ARCH_REGISTRY))
            raise ValueError(
                f"Unknown arch '{name}'. Available: {available}"
            )
        log.info(f"Instantiating architecture from registry: '{name}'")
        model = factory()
        return self._load_state(model, state_dict)

    @staticmethod
    def _load_state(model, state_dict: dict):
        import torch
        try:
            model.load_state_dict(state_dict, strict=True)
            log.info("State dict loaded (strict=True).")
        except RuntimeError as e:
            log.warning(f"Strict load failed ({e}); retrying with strict=False …")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                log.warning(f"  Missing keys  ({len(missing)}): {missing[:5]} …")
            if unexpected:
                log.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
        return model



# ──────────────────────────────────────────────────────────────────────────────
# Ultralytics / YOLOv8 converter
# ──────────────────────────────────────────────────────────────────────────────

def _is_ultralytics_checkpoint(path: Path) -> bool:
    """
    Peek inside a .pt file to see if it was saved by Ultralytics.
    Checks for tell-tale keys without fully loading the model.
    """
    try:
        import torch
        # weights_only=False needed; we only look at top-level keys in the dict
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            # Could be a full nn.Module saved directly — check class name
            cls_name = type(ckpt).__module__ + "." + type(ckpt).__qualname__
            return "ultralytics" in cls_name.lower()
        # Ultralytics checkpoints always have these keys
        ultralytics_keys = {"model", "train_args", "ema", "epoch", "optimizer"}
        return bool(ultralytics_keys & set(ckpt.keys()))
    except Exception:
        return False


class UltralyticsConverter(BaseConverter):
    """
    YOLOv8 / YOLOv5 (Ultralytics) .pt  →  ONNX via ultralytics.export().

    Why a separate converter?
    ─────────────────────────
    Ultralytics saves full model objects that embed custom Python classes
    (DetectionModel, C2f, SPPF …).  torch.load() only works if the exact
    same ultralytics version is installed, and the resulting object is NOT
    a plain nn.Module — it has Ultralytics-specific wrappers, post-processing
    fused into the graph, and a non-standard forward() signature.

    The safest, most reliable path is to let Ultralytics export to ONNX
    itself (it handles pre/post-processing, opset, dynamic axes, and
    input name conventions correctly), then pass the ONNX to TensorRT.

    Input shape
    ───────────
    YOLOv8 models accept any multiple-of-32 square resolution.  The default
    is 640×640.  If cfg.input_specs is empty the converter uses 640×640;
    otherwise it reads the first spec's HW dims.
    """

    # Default imgsz when no spec provided
    DEFAULT_IMGSZ = 640

    def to_onnx(self, tmp_dir: Path) -> Path:
        self._check_import(
            "ultralytics",
            "pip install ultralytics",
        )
        from ultralytics import YOLO  # type: ignore

        # ── resolve target image size ──────────────────────────────────────
        imgsz = self._resolve_imgsz()
        log.info(f"Loading Ultralytics model from {self.cfg.source_path} …")
        yolo = YOLO(str(self.cfg.source_path))

        # ── export to ONNX ─────────────────────────────────────────────────
        export_dir = tmp_dir / "yolo_export"
        export_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Exporting YOLOv8 → ONNX  (imgsz={imgsz}, opset=17) …")
        exported = yolo.export(
            format="onnx",
            imgsz=imgsz,
            opset=17,
            dynamic=True,        # dynamic batch axis
            simplify=True,       # onnx-simplifier if available
            half=False,          # keep fp32; TRT handles precision
            int8=False,
            project=str(export_dir),
            name="model",
        )
        onnx_path = Path(exported)
        if not onnx_path.exists():
            # Some ultralytics versions return the dir, not the file
            candidates = list(export_dir.rglob("*.onnx"))
            if not candidates:
                raise RuntimeError(
                    f"Ultralytics ONNX export did not produce a .onnx file "
                    f"in {export_dir}."
                )
            onnx_path = candidates[0]

        log.info(f"ONNX exported to {onnx_path}")

        # ── back-fill input specs from the ONNX graph ─────────────────────
        # (ultralytics names the input 'images' with shape [B,3,H,W])
        self._sync_specs_from_onnx(onnx_path)

        return onnx_path

    # ── helpers ───────────────────────────────────────────────────────────
    def _resolve_imgsz(self) -> int:
        """
        Return the image size (single int, square) to pass to yolo.export().
        Priority: cfg.input_specs H dim → DEFAULT_IMGSZ.
        """
        if self.cfg.input_specs:
            dims = self.cfg.input_specs[0].dims  # CHW
            if len(dims) == 3:
                h = dims[1]
                if h > 0 and h % 32 == 0:
                    return h
                log.warning(
                    f"Input height {h} is not a multiple of 32; "
                    f"rounding to {self.DEFAULT_IMGSZ}."
                )
        return self.DEFAULT_IMGSZ

    def _sync_specs_from_onnx(self, onnx_path: Path) -> None:
        """
        Read the ONNX graph and push input/output specs back into cfg so
        the pbtxt generator has the correct tensor names and shapes.
        """
        try:
            import onnx  # type: ignore
            model = onnx.load(str(onnx_path))

            # inputs
            new_in = []
            for inp in model.graph.input:
                t = inp.type.tensor_type
                dims = []
                for i, d in enumerate(t.shape.dim):
                    if i == 0: continue          # skip batch
                    dims.append(d.dim_value if d.dim_value > 0 else -1)
                dtype = _ONNX_DTYPE_TO_TRITON.get(t.elem_type, "TYPE_FP32")
                new_in.append(TensorSpec(name=inp.name, dims=dims, dtype=dtype))
            if new_in:
                self.cfg.input_specs = new_in
                log.info(f"Synced input specs from ONNX: {[(s.name, s.dims) for s in new_in]}")

            # outputs
            new_out = []
            for out in model.graph.output:
                t = out.type.tensor_type
                dims = []
                for i, d in enumerate(t.shape.dim):
                    if i == 0: continue
                    dims.append(d.dim_value if d.dim_value > 0 else -1)
                dtype = _ONNX_DTYPE_TO_TRITON.get(t.elem_type, "TYPE_FP32")
                new_out.append(TensorSpec(name=out.name, dims=dims, dtype=dtype))
            if new_out:
                self.cfg.output_specs = new_out
                log.info(f"Synced output specs from ONNX: {[(s.name, s.dims) for s in new_out]}")

        except Exception as e:
            log.warning(f"Could not sync specs from ONNX graph: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Converter registry
# ──────────────────────────────────────────────────────────────────────────────
CONVERTER_REGISTRY: Dict[str, type] = {
    ".onnx": OnnxConverter,
    ".keras": KerasConverter,
    ".h5": KerasConverter,
    ".pt": PytorchConverter,
    ".pth": PytorchConverter,
}


def get_converter(cfg: ConversionConfig) -> BaseConverter:
    """
    Return the right converter for cfg.source_path.

    For .pt files the choice is ambiguous: Ultralytics YOLO files look
    identical to regular PyTorch checkpoints at the filesystem level.
    We peek inside the file to decide:
      • Ultralytics checkpoint  → UltralyticsConverter
      • Anything else           → PytorchConverter
    """
    suffix = cfg.source_path.suffix.lower()

    if suffix in (".pt", ".pth"):
        if _is_ultralytics_checkpoint(cfg.source_path):
            log.info(
                f"Detected Ultralytics checkpoint → using UltralyticsConverter"
            )
            return UltralyticsConverter(cfg)
        return PytorchConverter(cfg)

    cls = CONVERTER_REGISTRY.get(suffix)
    if cls is None:
        raise ValueError(
            f"Unsupported format '{suffix}'. "
            f"Supported: {list(CONVERTER_REGISTRY)}"
        )
    return cls(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT builder
# ──────────────────────────────────────────────────────────────────────────────
class TensorRTBuilder:
    """Wraps trtexec or the tensorrt Python API to build a .engine file."""

    def __init__(self, cfg: ConversionConfig):
        self.cfg = cfg

    # ── public entry point ─────────────────────────────────────────────────
    def build(self, onnx_path: Path) -> Path:
        """Try Python API first, fall back to trtexec CLI."""
        try:
            import tensorrt as trt  # type: ignore
            log.info(f"TensorRT {trt.__version__} found – using Python API.")
            return self._build_python_api(onnx_path, trt)
        except ImportError:
            log.warning("tensorrt Python package not found – trying trtexec CLI.")
            return self._build_trtexec_cli(onnx_path)

    # ── Python API path ────────────────────────────────────────────────────
    def _build_python_api(self, onnx_path: Path, trt) -> Path:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        log.info(f"Parsing ONNX: {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("ONNX parse errors:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.cfg.workspace_mb * (1 << 20),
        )

        # Precision flags
        if self.cfg.precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            log.info("FP16 precision enabled.")
        elif self.cfg.precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            log.warning("INT8 selected – you must supply a calibrator for accuracy.")
        else:
            log.info("FP32 precision (default).")

        # Optimisation profiles (dynamic shapes)
        profile = builder.create_optimization_profile()
        for spec in self.cfg.input_specs:
            min_shape = [1] + spec.dims
            opt_shape = [max(1, self.cfg.max_batch_size // 2)] + spec.dims
            max_shape = [self.cfg.max_batch_size] + spec.dims
            profile.set_shape(spec.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        log.info("Building TensorRT engine (this may take several minutes) …")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed.")

        engine_path = self.cfg.engine_path
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(serialized)
        log.info(f"Engine saved → {engine_path}")
        return engine_path

    # ── trtexec CLI path ───────────────────────────────────────────────────
    def _build_trtexec_cli(self, onnx_path: Path) -> Path:
        trtexec = shutil.which("trtexec")
        if trtexec is None:
            raise EnvironmentError(
                "Neither the tensorrt Python package nor trtexec are available. "
                "Install TensorRT: https://developer.nvidia.com/tensorrt"
            )

        engine_path = self.cfg.engine_path
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--workspace={self.cfg.workspace_mb}",
            f"--device={self.cfg.device}",
        ]

        if self.cfg.precision == "fp16":
            cmd.append("--fp16")
        elif self.cfg.precision == "int8":
            cmd.append("--int8")

        # Dynamic shape profiles
        min_shapes, opt_shapes, max_shapes = [], [], []
        for s in self.cfg.input_specs:
            dims = "x".join(str(d) for d in s.dims)
            min_shapes.append(f"{s.name}:1x{dims}")
            opt_shapes.append(f"{s.name}:{max(1, self.cfg.max_batch_size // 2)}x{dims}")
            max_shapes.append(f"{s.name}:{self.cfg.max_batch_size}x{dims}")

        cmd += [
            f"--minShapes={','.join(min_shapes)}",
            f"--optShapes={','.join(opt_shapes)}",
            f"--maxShapes={','.join(max_shapes)}",
        ]

        log.info("Running: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"trtexec failed:\n{result.stderr}")

        log.info(f"Engine saved → {engine_path}")
        return engine_path


# ──────────────────────────────────────────────────────────────────────────────
# Triton pbtxt generator
# ──────────────────────────────────────────────────────────────────────────────
DTYPE_MAP = {
    "fp32": "TYPE_FP32",
    "fp16": "TYPE_FP16",
    "int8":  "TYPE_INT8",
    "int32": "TYPE_INT32",
    "int64": "TYPE_INT64",
    "bool":  "TYPE_BOOL",
    "uint8": "TYPE_UINT8",
}


def _tensor_block(label: str, specs: List[TensorSpec], index_offset: int = 0) -> str:
    lines = []
    for i, s in enumerate(specs):
        dims_str = ", ".join(str(d) for d in s.triton_dims())
        lines.append(
            f"{label} {{\n"
            f"  name: \"{s.name}\"\n"
            f"  data_type: {s.dtype}\n"
            f"  dims: [ {dims_str} ]\n"
            f"}}"
        )
    return "\n".join(lines)


def generate_pbtxt(cfg: ConversionConfig) -> str:
    """Return the full config.pbtxt content as a string."""

    # Infer output specs if none provided
    output_specs = cfg.output_specs or [
        TensorSpec(name="output", dims=[-1], dtype="TYPE_FP32")
    ]

    lines = [
        f'name: "{cfg.model_name}"',
        'backend: "tensorrt"',
        f"max_batch_size: {cfg.max_batch_size}",
        "",
        _tensor_block("input", cfg.input_specs),
        "",
        _tensor_block("output", output_specs),
        "",
        "dynamic_batching {",
        f"  preferred_batch_size: [ {cfg.max_batch_size} ]",
        f"  max_queue_delay_microseconds: 100",
        "}",
        "",
        "instance_group [",
        "  {",
        "    kind: KIND_GPU",
        f"    gpus: [ {cfg.device} ]",
        "    count: 1",
        "  }",
        "]",
        "",
        "optimization {",
        "  cuda {",
        "    graphs: true",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# ONNX output-shape introspection (optional, best-effort)
# ──────────────────────────────────────────────────────────────────────────────
def _infer_output_specs_from_onnx(onnx_path: Path) -> List[TensorSpec]:
    """Try to parse output tensor names/shapes from the ONNX graph."""
    try:
        import onnx  # type: ignore
        import onnx.numpy_helper as nph  # type: ignore

        _ONNX_TO_TRITON = {
            1: "TYPE_FP32",
            2: "TYPE_UINT8",
            3: "TYPE_INT8",
            5: "TYPE_INT32",
            6: "TYPE_INT32",
            7: "TYPE_INT64",
            9: "TYPE_BOOL",
            10: "TYPE_FP16",
            11: "TYPE_FP64",
            12: "TYPE_UINT32",
        }

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        specs = []
        for out in model.graph.output:
            shape = out.type.tensor_type.shape
            dims = []
            for i, d in enumerate(shape.dim):
                if i == 0:
                    continue  # skip batch dim
                dims.append(d.dim_value if d.dim_value > 0 else -1)
            dtype = _ONNX_TO_TRITON.get(
                out.type.tensor_type.elem_type, "TYPE_FP32"
            )
            specs.append(TensorSpec(name=out.name, dims=dims or [-1], dtype=dtype))
        log.info(f"Auto-detected {len(specs)} output(s) from ONNX graph.")
        return specs
    except Exception as e:
        log.warning(f"Could not auto-detect output specs: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────────────
class ConversionPipeline:
    """Ties together: format converter → TRT builder → pbtxt writer."""

    def __init__(self, cfg: ConversionConfig):
        self.cfg = cfg

    def run(self) -> Tuple[Path, Path]:
        """Execute the full pipeline.  Returns (engine_path, pbtxt_path)."""
        cfg = self.cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 0. Resolve input shapes if not supplied ──────────────────────────────
        if not cfg.input_specs:
            log.info("No --input-shapes given; attempting auto-detection …")
            detector = ShapeDetector(cfg.source_path)
            cfg.input_specs = detector.detect(
                input_name=cfg.image_input_name,
                # Preserve model graph names unless user explicitly typed one
                user_supplied_name=bool(
                    cfg.image_input_name and cfg.image_input_name != "input"
                ),
            )

        with tempfile.TemporaryDirectory(prefix="trt_convert_") as tmp:
            tmp_dir = Path(tmp)

            # 1. Convert source format → ONNX
            converter = get_converter(cfg)
            onnx_path = converter.to_onnx(tmp_dir)

            # 1b. Resolve any remaining dynamic dims produced by the ONNX export
            if any(-1 in s.dims for s in cfg.input_specs):
                detector = ShapeDetector(cfg.source_path)
                cfg.input_specs = _resolve_dynamic_dims(cfg.input_specs, detector)

            # 2. Auto-detect output specs if not supplied
            if not cfg.output_specs:
                cfg.output_specs = _infer_output_specs_from_onnx(onnx_path)

            # 3. Copy ONNX alongside the engine (useful for debugging)
            shutil.copy2(onnx_path, cfg.onnx_path)
            log.info(f"ONNX cached at {cfg.onnx_path}")

            # 4. Build TensorRT engine
            builder = TensorRTBuilder(cfg)
            engine_path = builder.build(onnx_path)

        # 5. Write Triton config.pbtxt
        pbtxt_content = generate_pbtxt(cfg)
        cfg.pbtxt_path.write_text(pbtxt_content, encoding="utf-8")
        log.info(f"config.pbtxt written → {cfg.pbtxt_path}")

        # 6. Summary
        self._print_summary(engine_path)
        return engine_path, cfg.pbtxt_path

    def _print_summary(self, engine_path: Path) -> None:
        cfg = self.cfg
        log.info("=" * 60)
        log.info("Conversion complete!")
        log.info(f"  Model name : {cfg.model_name}")
        log.info(f"  Precision  : {cfg.precision.upper()}")
        log.info(f"  Max batch  : {cfg.max_batch_size}")
        log.info(f"  Engine     : {engine_path}")
        log.info(f"  pbtxt      : {cfg.pbtxt_path}")
        log.info("")
        log.info("Triton model repository structure:")
        for p in sorted(cfg.output_dir.rglob("*")):
            rel = p.relative_to(cfg.output_dir)
            indent = "  " + "  " * (len(rel.parts) - 1)
            log.info(f"{indent}{p.name}")
        log.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert .keras/.onnx/.pt/.pth → TensorRT .engine + Triton pbtxt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required (but skipped when --list-archs is used)
    p.add_argument("--model", default=None,
                   help="Path to source model file.")
    p.add_argument("--output-dir", default=None,
                   help="Triton model repository root for this model "
                        "(e.g. triton_repo/my_model).")

    # Shape group – one of these must be given (or neither for full auto-detect)
    shape_group = p.add_mutually_exclusive_group()
    shape_group.add_argument("--input-shapes", nargs="+", default=[],
                   metavar="SPEC",
                   help="Explicit input specs: 'name:CxHxW' (batch excluded). "
                        "Example: images:3x640x640")
    shape_group.add_argument("--image-input", nargs="?", const="input",
                   metavar="INPUT_NAME",
                   help="Trigger image-shape auto-detection. Optionally "
                        "set the tensor name (default: 'input'). "
                        "Reads shape from model file; falls back to "
                        "an interactive menu of common resolutions.")

    # Optional
    p.add_argument("--model-name", default=None,
                   help="Triton model name (defaults to stem of --model).")
    p.add_argument("--output-shapes", nargs="*", default=[],
                   help="Output tensor specs 'name:d0xd1x…' "
                        "(auto-detected from ONNX if omitted).")
    p.add_argument("--max-batch", type=int, default=1,
                   help="Maximum batch size for TRT optimisation profile.")
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32",
                   help="TensorRT precision mode.")
    p.add_argument("--workspace-mb", type=int, default=4096,
                   help="TRT builder workspace in MiB.")
    p.add_argument("--device", type=int, default=0,
                   help="GPU device index.")
    p.add_argument("--triton-version", type=int, default=1,
                   help="Triton model version (subfolder number).")
    # Architecture (for state-dict-only PyTorch checkpoints)
    p.add_argument("--arch", default=None,
                   metavar="NAME",
                   help="Built-in architecture name for state-dict checkpoints. "
                        "Run with --list-archs to see all options. "
                        "Example: cyclegan_discriminator, resnet50")
    p.add_argument("--model-module", default=None,
                   metavar="FILE::CLASS",
                   help="Load architecture from an external Python file. "
                        "Format: path/to/model.py::ClassName")
    p.add_argument("--list-archs", action="store_true",
                   help="Print all built-in architecture names and exit.")
    p.add_argument("--input-dtype", default="fp32",
                   choices=list(DTYPE_MAP),
                   help="Data type for all inputs (used in pbtxt).")
    p.add_argument("--output-dtype", default="fp32",
                   choices=list(DTYPE_MAP),
                   help="Data type for all outputs (used in pbtxt).")

    return p.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> ConversionConfig:
    if not args.model:
        raise SystemExit("error: --model is required (unless using --list-archs)")
    if not args.output_dir:
        raise SystemExit("error: --output-dir is required (unless using --list-archs)")
    source = Path(args.model).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Model not found: {source}")

    model_name = args.model_name or source.stem
    output_dir = Path(args.output_dir).expanduser().resolve() / model_name

    in_dtype  = DTYPE_MAP[args.input_dtype]
    out_dtype = DTYPE_MAP[args.output_dtype]

    # --input-shapes given explicitly
    input_specs = [TensorSpec.from_str(s) for s in (args.input_shapes or [])]
    for s in input_specs:
        s.dtype = in_dtype

    output_specs = [
        TensorSpec.from_str(s) for s in (args.output_shapes or [])
    ]
    for s in output_specs:
        s.dtype = out_dtype

    # --image-input sets auto-detect mode (input_specs left empty for pipeline)
    image_input_name: Optional[str] = None
    if not input_specs:
        # Either --image-input was given, or neither flag was given (full auto)
        image_input_name = getattr(args, "image_input", None) or "input"

    return ConversionConfig(
        source_path=source,
        output_dir=output_dir,
        model_name=model_name,
        input_specs=input_specs,        # empty → pipeline will auto-detect
        output_specs=output_specs,
        max_batch_size=args.max_batch,
        precision=args.precision,
        workspace_mb=args.workspace_mb,
        device=args.device,
        triton_version=args.triton_version,
        image_input_name=image_input_name,
        arch_name=getattr(args, 'arch', None),
        model_module=getattr(args, 'model_module', None),
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    if getattr(args, 'list_archs', False):
        print("Built-in architecture registry:")
        for name in sorted(ARCH_REGISTRY):
            print(f"  {name}")
        return
    cfg  = build_config_from_args(args)
    pipeline = ConversionPipeline(cfg)
    pipeline.run()


# ──────────────────────────────────────────────────────────────────────────────
# Public Python API  (import-friendly, no CLI needed)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConvertResult:
    """
    Returned by convert_models() for every model in the input list.

    Fields
    ------
    model_path  : original source file
    engine_path : produced .plan file (None on failure)
    pbtxt_path  : produced config.pbtxt (None on failure)
    input_specs : resolved input tensor specs (after auto-detection)
    output_specs: resolved output tensor specs
    success     : True if conversion finished without exception
    error       : exception message if success is False
    """
    model_path:   Path
    engine_path:  Optional[Path]
    pbtxt_path:   Optional[Path]
    input_specs:  List[TensorSpec]
    output_specs: List[TensorSpec]
    success:      bool
    error:        Optional[str] = None

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"[{status}] {self.model_path.name}"
            + (f"  →  {self.engine_path}" if self.success else f"  ERROR: {self.error}")
        )


def convert_models(
    models: List[str | Path],
    output_dir: str | Path,
    *,
    # ── precision / hardware ───────────────────────────────────────────────
    precision: str = "fp32",            # "fp32" | "fp16" | "int8"
    max_batch_size: int = 1,
    workspace_mb: int = 4096,
    device: int = 0,
    triton_version: int = 1,
    # ── shape hints (applied to all models; auto-detected when omitted) ───
    input_specs:  Optional[List[TensorSpec]] = None,
    output_specs: Optional[List[TensorSpec]] = None,
    # ── behaviour flags ───────────────────────────────────────────────────
    skip_on_error: bool = True,         # False → re-raise the first failure
    non_interactive: bool = False,      # True → never prompt; use 3×224×224
    show_summary: bool = True,
) -> List[ConvertResult]:
    """
    Convert a list of model files to TensorRT engines and Triton pbtxt configs.

    All shapes are auto-detected (ONNX graph, Keras .input_shape, torchinfo,
    or the interactive image-shape menu).  Pass ``input_specs`` / ``output_specs``
    to override for every model in the batch, or leave them as *None* to let
    each model be inspected individually.

    Parameters
    ----------
    models : list of paths
        Source model files (.onnx, .keras, .h5, .pt, .pth).
    output_dir : str | Path
        Root of the Triton model repository.  Each model gets its own
        sub-folder: ``<output_dir>/<model_stem>/``.
    precision : {"fp32", "fp16", "int8"}
        TensorRT precision mode applied to all models.
    max_batch_size : int
        Maximum batch size for the TRT optimisation profile.
    workspace_mb : int
        TRT builder workspace size in MiB.
    device : int
        GPU device index.
    triton_version : int
        Version number folder created inside each model directory.
    input_specs : list[TensorSpec] | None
        If supplied, these specs override auto-detection for every model.
        Useful when you know the shape is identical across models.
    output_specs : list[TensorSpec] | None
        Same as input_specs but for outputs.
    skip_on_error : bool
        When *True* (default) a failing model is recorded and the batch
        continues.  When *False* the first failure raises immediately.
    non_interactive : bool
        Force non-interactive mode (default 3×224×224 fallback) even when
        stdin is a TTY.  Handy for automation scripts.
    show_summary : bool
        Print a rich summary table after all conversions finish.

    Returns
    -------
    list[ConvertResult]
        One result per model, in the same order as the input list.

    Examples
    --------
    >>> from convert_to_trt import convert_models
    >>>
    >>> # Minimal – fully automatic
    >>> results = convert_models(
    ...     models=["yolov8n.onnx", "resnet50.pt", "efficientnet.keras"],
    ...     output_dir="./triton_repo",
    ...     precision="fp16",
    ...     max_batch_size=8,
    ... )
    >>> for r in results:
    ...     print(r)
    >>>
    >>> # Override shape for every model (e.g. same input across all)
    >>> from convert_to_trt import TensorSpec
    >>> results = convert_models(
    ...     models=["model_a.onnx", "model_b.onnx"],
    ...     output_dir="./triton_repo",
    ...     input_specs=[TensorSpec("input", [3, 640, 640])],
    ...     output_specs=[TensorSpec("output", [80, 85])],
    ...     precision="fp16",
    ... )
    """

    # ── normalise paths ────────────────────────────────────────────────────
    model_paths = [Path(m).expanduser().resolve() for m in models]
    out_root    = Path(output_dir).expanduser().resolve()

    # ── patch sys.stdin for non-interactive mode ───────────────────────────
    # We monkey-patch isatty so ShapeDetector._prompt_image_shape takes
    # the silent fallback path instead of hanging waiting for input.
    _orig_isatty = getattr(sys.stdin, "isatty", lambda: True)
    if non_interactive:
        sys.stdin.isatty = lambda: False  # type: ignore[method-assign]

    results: List[ConvertResult] = []

    try:
        for model_path in model_paths:
            log.info("─" * 60)
            log.info(f"Processing: {model_path.name}")

            result = _convert_single(
                model_path=model_path,
                out_root=out_root,
                precision=precision,
                max_batch_size=max_batch_size,
                workspace_mb=workspace_mb,
                device=device,
                triton_version=triton_version,
                input_specs_override=input_specs,
                output_specs_override=output_specs,
                skip_on_error=skip_on_error,
            )
            results.append(result)

    finally:
        # Restore stdin.isatty regardless of exceptions
        if non_interactive:
            sys.stdin.isatty = _orig_isatty  # type: ignore[method-assign]

    if show_summary:
        _print_batch_summary(results)

    return results


# ── internal helpers ───────────────────────────────────────────────────────────

def _convert_single(
    *,
    model_path: Path,
    out_root: Path,
    precision: str,
    max_batch_size: int,
    workspace_mb: int,
    device: int,
    triton_version: int,
    input_specs_override: Optional[List[TensorSpec]],
    output_specs_override: Optional[List[TensorSpec]],
    skip_on_error: bool,
) -> ConvertResult:
    """Run the full pipeline for one model file, returning a ConvertResult."""

    # Deep-copy specs so per-model auto-detection doesn't bleed across models
    import copy
    in_specs  = copy.deepcopy(input_specs_override)  or []
    out_specs = copy.deepcopy(output_specs_override) or []

    # ── build ConversionConfig ─────────────────────────────────────────────
    model_name = model_path.stem
    cfg = ConversionConfig(
        source_path=model_path,
        output_dir=out_root / model_name,
        model_name=model_name,
        input_specs=in_specs,       # empty list → pipeline auto-detects
        output_specs=out_specs,
        max_batch_size=max_batch_size,
        precision=precision,
        workspace_mb=workspace_mb,
        device=device,
        triton_version=triton_version,
        image_input_name="input",   # default name used during auto-detection
    )

    try:
        engine_path, pbtxt_path = ConversionPipeline(cfg).run()
        return ConvertResult(
            model_path=model_path,
            engine_path=engine_path,
            pbtxt_path=pbtxt_path,
            input_specs=cfg.input_specs,
            output_specs=cfg.output_specs,
            success=True,
        )

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        log.error(f"Failed to convert {model_path.name}: {msg}")
        if not skip_on_error:
            raise
        return ConvertResult(
            model_path=model_path,
            engine_path=None,
            pbtxt_path=None,
            input_specs=cfg.input_specs,
            output_specs=cfg.output_specs,
            success=False,
            error=msg,
        )


def _print_batch_summary(results: List[ConvertResult]) -> None:
    """Print a tidy summary table after a batch conversion."""
    ok  = [r for r in results if r.success]
    bad = [r for r in results if not r.success]

    log.info("")
    log.info("═" * 70)
    log.info(f"  BATCH SUMMARY   {len(ok)} succeeded  /  {len(bad)} failed")
    log.info("═" * 70)

    for r in results:
        tick = "✓" if r.success else "✗"
        in_shapes  = " | ".join(
            f"{s.name}:{s.dims}" for s in r.input_specs
        ) or "—"
        out_shapes = " | ".join(
            f"{s.name}:{s.dims}" for s in r.output_specs
        ) or "—"

        log.info(f"  [{tick}] {r.model_path.name}")
        if r.success:
            log.info(f"       inputs : {in_shapes}")
            log.info(f"       outputs: {out_shapes}")
            log.info(f"       engine : {r.engine_path}")
        else:
            log.info(f"       error  : {r.error}")

    log.info("═" * 70)
    log.info("")


if __name__ == "__main__":
    main()