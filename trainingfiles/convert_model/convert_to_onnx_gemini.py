#!/usr/bin/env python3
"""
convert_to_onnx.py
==================
Convert .keras / .h5 / .pt / .pth  →  .onnx, with native Triton Inference Server deployment support.

Usage
-----
python convert_to_onnx.py --model model.pt  --output model.onnx
python convert_to_onnx.py --model model.keras --output out/
python convert_to_onnx.py --model model.pth --arch resnet50
python convert_to_onnx.py --model model.pt --output triton_repo/ --triton --model-name my_vision_model
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
log = logging.getLogger("convert_to_onnx")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
COMMON_IMAGE_SHAPES: List[Tuple[int, int, int]] = [
    (3, 224, 224), (3, 256, 256), (3, 320, 320), (3, 384, 384),
    (3, 416, 416), (3, 512, 512), (3, 608, 608), (3, 640, 640),
    (3, 768, 768), (3, 1024, 1024), (1, 224, 224), (1, 256, 256), (1, 512, 512),
]

_ONNX_TO_TRITON_DTYPE = {
    1: "TYPE_FP32", 2: "TYPE_UINT8", 3: "TYPE_INT8",
    6: "TYPE_INT32", 7: "TYPE_INT64", 9: "TYPE_BOOL",
    10: "TYPE_FP16", 11: "TYPE_FP64"
}


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class InputSpec:
    name: str
    dims: List[int]

@dataclass
class ConvertResult:
    source_path: Path
    onnx_path:   Optional[Path]
    input_specs: List[InputSpec]
    success:     bool
    error:       Optional[str] = None

    def __str__(self) -> str:
        mark = "✓" if self.success else "✗"
        detail = str(self.onnx_path) if self.success else self.error
        return f"[{mark}] {self.source_path.name}  →  {detail}"


# ──────────────────────────────────────────────────────────────────────────────
# Architecture registry
# ──────────────────────────────────────────────────────────────────────────────
def _torchvision_factory(name: str):
    def _make():
        import torchvision.models as M  # type: ignore
        ctor = getattr(M, name, None)
        if ctor is None:
            raise ValueError(f"torchvision has no model '{name}'.")
        return ctor(weights=None)
    return _make

def _make_cyclegan_resnet_generator(n_blocks: int = 9):
    import torch.nn as nn
    class ResnetBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3),
                nn.InstanceNorm2d(dim), nn.ReLU(True),
                nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3),
                nn.InstanceNorm2d(dim),
            )
        def forward(self, x): return x + self.block(x)

    ngf = 64
    layers = [
        nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
        nn.Conv2d(ngf,   ngf*2, 3, 2, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
        nn.Conv2d(ngf*2, ngf*4, 3, 2, 1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
    ]
    for _ in range(n_blocks): layers.append(ResnetBlock(ngf * 4))
    layers += [
        nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
        nn.ConvTranspose2d(ngf*2, ngf,   3, 2, 1, 1), nn.InstanceNorm2d(ngf),   nn.ReLU(True),
        nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh(),
    ]
    return nn.Sequential(*layers)

ARCH_REGISTRY: Dict[str, Any] = {
    "cyclegan_resnet_9": lambda: _make_cyclegan_resnet_generator(9),
    "resnet18":          _torchvision_factory("resnet18"),
    "resnet50":          _torchvision_factory("resnet50"),
    "yolov8":            None, # Handled directly by Ultralytics exporter
}


# ──────────────────────────────────────────────────────────────────────────────
# Shape detector
# ──────────────────────────────────────────────────────────────────────────────
class ShapeDetector:
    def __init__(self, path: Path):
        self.path   = path
        self.suffix = path.suffix.lower()

    def detect(self) -> List[InputSpec]:
        specs = None
        if self.suffix in (".keras", ".h5"):
            specs = self._from_keras()
        elif self.suffix in (".pt", ".pth"):
            specs = self._from_pytorch()

        if specs:
            log.info(f"Auto-detected inputs: {[(s.name, s.dims) for s in specs]}")
            return specs

        log.warning("Could not read input shapes. Falling back to image-shape probe.")
        return self._prompt(name="input")

    def _from_keras(self) -> Optional[List[InputSpec]]:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(self.path), compile=False)
            raw = model.inputs if hasattr(model, "inputs") else [model.input]
            specs = []
            for inp in raw:
                shape = inp.shape.as_list()[1:]
                dims  = self._nhwc_to_chw(shape) if len(shape) == 3 else shape
                dims  = [-1 if (d is None or d <= 0) else d for d in dims]
                specs.append(InputSpec(name=inp.name.split(":")[0], dims=dims))
            return specs or None
        except Exception as e:
            return None

    def _from_pytorch(self) -> Optional[List[InputSpec]]:
        try:
            import torch
            ckpt = torch.load(str(self.path), map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                for container in (ckpt.get("train_args"), ckpt.get("args")):
                    imgsz = getattr(container, "imgsz", None) if not isinstance(container, dict) else container.get("imgsz")
                    if imgsz:
                        h = w = int(imgsz) if isinstance(imgsz, (int, float)) else int(imgsz[0])
                        return [InputSpec(name="images", dims=[3, h, w])]
        except Exception:
            pass
        return None

    def _prompt(self, name: str) -> List[InputSpec]:
        if not sys.stdin.isatty():
            return [InputSpec(name=name, dims=[3, 224, 224])]
        print("\n" + "─" * 58 + "\nSelect input image shape:\n")
        for i, (c, h, w) in enumerate(COMMON_IMAGE_SHAPES, 1):
            print(f"  [{i:2d}] {c}×{h}×{w}")
        print("  [ c] custom\n" + "─" * 58)
        while True:
            raw = input("Enter choice: ").strip().lower()
            if raw == "c":
                raw_c = input("Enter C H W: ").split()
                return [InputSpec(name=name, dims=[int(x) for x in raw_c])]
            try:
                dims = list(COMMON_IMAGE_SHAPES[int(raw) - 1])
                return [InputSpec(name=name, dims=dims)]
            except (ValueError, IndexError):
                pass

    @staticmethod
    def _nhwc_to_chw(dims: list) -> list:
        if len(dims) == 3:
            h, w, c = dims
            return [c, h, w]
        return dims


# ──────────────────────────────────────────────────────────────────────────────
# Converters
# ──────────────────────────────────────────────────────────────────────────────
class BaseConverter(ABC):
    def __init__(self, path: Path, specs: List[InputSpec], opset: int = 17):
        self.path  = path
        self.specs = specs
        self.opset = opset

    @abstractmethod
    def convert(self, output_path: Path) -> Path: ...


class KerasConverter(BaseConverter):
    def convert(self, output_path: Path) -> Path:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        import tf2onnx

        model = tf.keras.models.load_model(str(self.path), compile=False)
        tf_specs = []
        for s in self.specs:
            shape = [None, s.dims[1], s.dims[2], s.dims[0]] if len(s.dims) == 3 else [None] + s.dims
            tf_specs.append(tf.TensorSpec(shape=shape, dtype=tf.float32, name=s.name))

        tf2onnx.convert.from_keras(model, input_signature=tf_specs, opset=self.opset, output_path=str(output_path))
        return output_path


class PytorchConverter(BaseConverter):
    def __init__(self, path: Path, specs: List[InputSpec], opset: int = 17, arch_name: Optional[str] = None, model_module: Optional[str] = None):
        super().__init__(path, specs, opset)
        self.arch_name    = arch_name
        self.model_module = model_module

    def convert(self, output_path: Path) -> Path:
        import torch
        ckpt = torch.load(str(self.path), map_location="cpu", weights_only=False)
        model = self._resolve(ckpt, torch)
        model.eval()

        inp = tuple([torch.zeros(1, *s.dims) for s in self.specs])
        if len(inp) == 1: inp = inp[0]

        in_names  = [s.name for s in self.specs]
        out_names = ["output"]
        dyn_axes  = {n: {0: "batch"} for n in in_names + out_names}

        torch.onnx.export(model, inp, str(output_path), opset_version=self.opset, input_names=in_names, output_names=out_names, dynamic_axes=dyn_axes)
        return output_path

    def _resolve(self, ckpt, torch):
        if isinstance(ckpt, torch.nn.Module): return ckpt
        sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
        if self.arch_name:
            factory = ARCH_REGISTRY.get(self.arch_name)
            model = factory()
            model.load_state_dict(sd, strict=False)
            return model
        raise ValueError("Requires --arch flag for raw state_dicts.")


class UltralyticsConverter(BaseConverter):
    def convert(self, output_path: Path) -> Path:
        from ultralytics import YOLO
        imgsz = self.specs[0].dims[1] if self.specs else 640
        yolo = YOLO(str(self.path))
        exported = yolo.export(format="onnx", imgsz=imgsz, opset=self.opset, dynamic=True, simplify=True)
        src = Path(exported)
        if src.is_dir(): src = list(src.rglob("*.onnx"))[0]
        shutil.copy2(src, output_path)
        return output_path


def _is_ultralytics(path: Path) -> bool:
    try:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            return "ultralytics" in (type(ckpt).__module__ + type(ckpt).__qualname__).lower()
        return bool({"model", "train_args", "ema"} & set(ckpt.keys()))
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Triton Infrastructure Builder
# ──────────────────────────────────────────────────────────────────────────────
def _generate_triton_config(onnx_file: Path, model_dir: Path, model_name: str, max_batch_size: int):
    """Parses the exported ONNX file to generate a precise config.pbtxt"""
    try:
        import onnx
    except ImportError:
        log.warning("Could not generate perfect config.pbtxt: 'onnx' package missing. Creating a skeleton config instead.")
        config_str = f'name: "{model_name}"\nplatform: "onnxruntime_onnx"\nmax_batch_size: {max_batch_size}\n'
        (model_dir / "config.pbtxt").write_text(config_str)
        return

    model = onnx.load(str(onnx_file))

    def _parse_io(io_list):
        configs = []
        for node in io_list:
            name = node.name
            tensor_type = node.type.tensor_type
            dtype = _ONNX_TO_TRITON_DTYPE.get(tensor_type.elem_type, "TYPE_FP32")
            
            dims = []
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(str(d.dim_value))
                else:
                    dims.append("-1")
            
            # Triton strips the batch dimension from the config if max_batch_size > 0
            if max_batch_size > 0 and len(dims) > 0:
                dims = dims[1:]
                
            dims_str = "[ " + ", ".join(dims) + " ]"
            configs.append(f'  {{\n    name: "{name}"\n    data_type: {dtype}\n    dims: {dims_str}\n  }}')
        return configs

    inps = _parse_io(model.graph.input)
    outs = _parse_io(model.graph.output)

    lines = [
        f'name: "{model_name}"',
        'platform: "onnxruntime_onnx"',
        f'max_batch_size: {max_batch_size}',
        '',
        'dynamic_batching {',
        '  max_queue_delay_microseconds: 100',
        '}',
        ''
    ]
    
    for i in inps:
        lines.append('input [\n' + i + '\n]')
    for o in outs:
        lines.append('output [\n' + o + '\n]')

    config_path = model_dir / "config.pbtxt"
    config_path.write_text("\n".join(lines))
    log.info(f"Triton config generated at: {config_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Core convert() function
# ──────────────────────────────────────────────────────────────────────────────
def convert(
    model:        str | Path,
    output:       str | Path,
    *,
    input_specs:  Optional[List[InputSpec]] = None,
    opset:        int = 17,
    arch_name:    Optional[str] = None,
    model_module: Optional[str] = None,
    triton:       bool = False,
    triton_name:  Optional[str] = None,
    triton_batch: int = 8,
) -> Path:
    src    = Path(model).expanduser().resolve()
    dst    = Path(output).expanduser().resolve()
    suffix = src.suffix.lower()

    if not src.exists():
        raise FileNotFoundError(f"Model not found: {src}")

    specs = input_specs or ShapeDetector(src).detect()

    # Determine paths based on whether we are building a Triton repo
    if triton:
        model_name = triton_name or src.stem
        # If the output path has an extension (like .onnx), use its parent directory as the base repo
        base_repo = dst if dst.suffix == "" else dst.parent
        triton_model_dir = base_repo / model_name
        version_dir = triton_model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        final_dst = version_dir / "model.onnx"
    else:
        if dst.is_dir(): dst = dst / (src.stem + ".onnx")
        final_dst = dst
        final_dst.parent.mkdir(parents=True, exist_ok=True)

    # Pick converter
    if suffix in (".keras", ".h5"):
        converter = KerasConverter(src, specs, opset)
    elif suffix in (".pt", ".pth"):
        if _is_ultralytics(src):
            converter = UltralyticsConverter(src, specs, opset)
        else:
            converter = PytorchConverter(src, specs, opset, arch_name, model_module)
    else:
        raise ValueError("Unsupported format.")

    onnx_path = converter.convert(final_dst)
    log.info(f"Done converting: {src.name}  →  {onnx_path}")

    # Generate config.pbtxt if Triton is requested
    if triton:
        _generate_triton_config(onnx_path, triton_model_dir, model_name, triton_batch)
        log.info(f"Triton model repository is ready at: {triton_model_dir}")

    return onnx_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Convert models to ONNX and optionally create a Triton repo.")
    p.add_argument("--model",      required=True, help="Source model file.")
    p.add_argument("--output",     default=None, help="Destination .onnx file or directory.")
    p.add_argument("--opset",      type=int, default=17)
    p.add_argument("--arch",       default=None)
    
    # Triton specific arguments
    p.add_argument("--triton", action="store_true", help="Generate Triton model repository structure.")
    p.add_argument("--model-name", default=None, help="Name for the Triton model (defaults to source filename).")
    p.add_argument("--max-batch-size", type=int, default=8, help="Max batch size for Triton config.pbtxt.")
    
    return p.parse_args()


def main():
    args = _parse_args()
    src = Path(args.model)
    dst = Path(args.output) if args.output else src.with_suffix(".onnx")

    convert(
        src, dst,
        opset=args.opset,
        arch_name=args.arch,
        triton=args.triton,
        triton_name=args.model_name,
        triton_batch=args.max_batch_size
    )

if __name__ == "__main__":
    main()