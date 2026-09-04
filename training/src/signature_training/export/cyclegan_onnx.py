#!/usr/bin/env python3
"""
export_cyclegan_onnx.py
=======================
Script to export CycleGAN generator checkpoints (.pth) to ONNX.

This script imports the ResNet-9-blocks generator architecture from the cloned
pytorch-CycleGAN-and-pix2pix repository.

Usage
-----
    # Export a single generator (default: G_B, the denoiser)
    python export_cyclegan_onnx.py \
        --checkpoint ../convert_model/original_model/latest_net_G_B.pth \
        --output     ../convert_model/original_model/latest_net_G_B.onnx

Requirements
------------
    - pytorch-CycleGAN-and-pix2pix must be cloned in the parent directory.
    - pip install torch>=2.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Import Architecture from CycleGAN Repo
# ─────────────────────────────────────────────────────────────────────────────

# Add the cloned repo to sys.path so we can import from it
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pytorch-CycleGAN-and-pix2pix"))
if not os.path.exists(repo_path):
    print(f"❌ Error: Could not find the pytorch-CycleGAN-and-pix2pix repository at {repo_path}", file=sys.stderr)
    print("Please clone it first: git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, repo_path)

try:
    from models.networks import define_G
except ImportError as e:
    print(f"❌ Error importing define_G from CycleGAN repo: {e}", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Export logic
# ─────────────────────────────────────────────────────────────────────────────

def export_cyclegan_to_onnx(checkpoint_path: str, output_path: str,
                             image_size: int = 224, device: str = "cpu") -> None:
    print(f"Loading weights from {checkpoint_path} ...")

    # 1. Build the generator blueprint using the repo's function
    net = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG='resnet_9blocks',
        norm='instance',
        use_dropout=False,
        init_type='normal',
        init_gain=0.02
    )

    # 2. Load the raw state dict
    map_location = torch.device(device)
    state_dict = torch.load(checkpoint_path, map_location=map_location)

    # 3. Clean up multi-GPU "module." prefixes
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 4. Load weights
    net.load_state_dict(clean_state)
    net.eval()
    net.to(map_location)

    # 5. Dummy input — locks input shape into the ONNX graph
    dummy = torch.randn(1, 3, image_size, image_size, device=map_location)

    # 6. Export
    print(f"Exporting to {output_path} ...")
    torch.onnx.export(
        net,
        dummy,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"✅ Export complete → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export CycleGAN generator .pth → .onnx (uses cloned CycleGAN repo)",
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to the .pth generator checkpoint")
    p.add_argument("--output", required=True,
                   help="Output .onnx file path")
    p.add_argument("--image-size", type=int, default=224,
                   help="Input image size (default: 224)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for export (default: cpu)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"❌ Checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)

    export_cyclegan_to_onnx(
        checkpoint_path=str(ckpt),
        output_path=args.output,
        image_size=args.image_size,
        device=args.device,
    )
