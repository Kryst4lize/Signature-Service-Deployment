"""CycleGAN generator checkpoint (.pth) -> ONNX.

The architecture comes from the cloned upstream repo's `define_G`, not from a
local reimplementation. That is deliberate. The two converters this repo used to
carry both defined their own ResNet generator, and both got the state_dict key
layout wrong: upstream wraps the network in an `nn.Sequential` stored as
`self.model`, so every key is `model.N....`, and upstream's ResnetBlock stores
its layers under `conv_block` where the local copies used `block`.

`load_state_dict(..., strict=False)` then reported 48 missing and 48 unexpected
keys — and returned a randomly initialised generator, which exports cleanly and
produces garbage. Calling upstream's own factory removes that entire class of
failure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def export(
    checkpoint: str | Path,
    onnx_path: str | Path,
    repo_path: str | Path,
    image_size: int = 224,
    opset: int = 14,
    device: str = "cpu",
) -> Path:
    import torch

    checkpoint, onnx_path, repo_path = Path(checkpoint), Path(onnx_path), Path(repo_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not (repo_path / "models" / "networks.py").is_file():
        raise FileNotFoundError(f"CycleGAN repo not found at {repo_path}. Run `sigtrain setup`.")

    sys.path.insert(0, str(repo_path))
    # Resolved at runtime from the CycleGAN repo just added to sys.path; it is
    # cloned, not a declared dependency.
    from models.networks import define_G  # ty: ignore[unresolved-import]

    net = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="resnet_9blocks",
        norm="instance",
        use_dropout=False,
        init_type="normal",
        init_gain=0.02,
        gpu_ids=[],
    )

    # weights_only=True refuses to unpickle arbitrary objects; a generator
    # checkpoint is a plain tensor dict, so nothing legitimate needs the
    # unrestricted loader.
    state = torch.load(checkpoint, map_location=torch.device(device), weights_only=True)
    if hasattr(state, "_metadata"):
        del state._metadata
    # Checkpoints saved under DataParallel carry a "module." prefix.
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # strict=True on purpose: a silent partial load is exactly the bug that
    # made the previous converters export an untrained network.
    net.load_state_dict(state, strict=True)
    net.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)

    logger.info("Exporting %s -> %s", checkpoint.name, onnx_path.name)
    torch.onnx.export(
        net,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    logger.info(
        "Exported. Note the generator ends in Tanh, so its output is [-1, 1]; "
        "the serving side rescales with (x+1)/2."
    )
    return onnx_path
