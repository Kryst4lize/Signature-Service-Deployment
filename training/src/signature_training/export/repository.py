"""Assemble a complete Triton model repository.

This is the hand-off between the two halves of the repo. It writes

    <triton_repository>/
        yolov8s/            { config.pbtxt, 1/model.onnx }
        latest_net_G_B/     { config.pbtxt, 1/model.onnx }
        resnet50_extractor/ { config.pbtxt, 1/model.onnx }
        vgg16_extractor/    { config.pbtxt, 1/model.onnx }

directly into inference/triton/model_repository by default, so deployment is
`docker compose up` rather than a manual copy that has to get four directory
names and four tensor names right.

Models that have not been trained yet are skipped with a warning rather than
failing the run — exporting only the extractors is a normal thing to want.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..config import Config
from . import cyclegan_onnx, keras_onnx, triton_config, yolo_onnx

logger = logging.getLogger(__name__)

# Triton model name -> the file the pipeline produces it from.
DENOISER_NAME = "latest_net_G_B"
DETECTOR_NAME = "yolov8s"


def build(cfg: Config, only: list[str] | None = None) -> dict[str, Path]:
    models_dir = cfg.paths.resolve("models")
    onnx_dir = cfg.paths.resolve("onnx")
    repo_dir = cfg.paths.resolve("triton_repository")
    onnx_dir.mkdir(parents=True, exist_ok=True)

    known = {
        DETECTOR_NAME, DENOISER_NAME,
        *(f"{b}_extractor" for b in keras_onnx.INPUT_NAMES),
    }
    wanted = set(only) if only else None
    if wanted:
        unknown = wanted - known
        if unknown:
            raise ValueError(
                f"Unknown model name(s) {sorted(unknown)}. "
                f"Valid names: {sorted(known)}"
            )

    def selected(name: str) -> bool:
        return wanted is None or name in wanted

    exported: dict[str, Path] = {}

    # ── extractors ────────────────────────────────────────────────────────────
    # Filter BEFORE converting. Filtering the results still loaded both .keras
    # models into TensorFlow, traced them with tf2onnx and rewrote
    # artifacts/onnx/, even for `--only latest_net_G_B`.
    wanted_extractors = [
        b for b in keras_onnx.INPUT_NAMES if selected(f"{b}_extractor")
    ]
    exported.update(
        keras_onnx.convert_extractors(
            models_dir, onnx_dir, cfg, backbones=wanted_extractors
        )
    )

    # ── denoiser ──────────────────────────────────────────────────────────────
    if selected(DENOISER_NAME):
        ckpt = (
            cfg.paths.resolve("cyclegan_checkpoints")
            / cfg.cyclegan_train.name
            / "latest_net_G_B.pth"
        )
        if ckpt.is_file():
            exported[DENOISER_NAME] = cyclegan_onnx.export(
                ckpt,
                onnx_dir / f"{DENOISER_NAME}.onnx",
                repo_path=cfg.paths.resolve("cyclegan_repo"),
                image_size=cfg.export.image_size,
            )
        else:
            logger.warning("Skipping %s - %s not found", DENOISER_NAME, ckpt)

    # ── detector ──────────────────────────────────────────────────────────────
    if selected(DETECTOR_NAME):
        weights = models_dir / "yolov8s.pt"
        if weights.is_file():
            exported[DETECTOR_NAME] = yolo_onnx.export(
                weights,
                onnx_dir / f"{DETECTOR_NAME}.onnx",
                image_size=cfg.export.yolo_image_size,
                opset=cfg.export.opset,
                simplify_graph=cfg.export.simplify,
            )
        else:
            logger.warning("Skipping %s - %s not found", DETECTOR_NAME, weights)

    if not exported:
        raise RuntimeError(
            f"Nothing to export. Looked for extractors in {models_dir}, a "
            f"CycleGAN checkpoint under {cfg.paths.resolve('cyclegan_checkpoints')}, "
            f"and {models_dir / 'yolov8s.pt'}."
        )

    # ── stage into the repository ─────────────────────────────────────────────
    for model_name, onnx_path in exported.items():
        model_dir = repo_dir / model_name
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, version_dir / "model.onnx")
        triton_config.write(
            onnx_path,
            model_dir,
            model_name,
            max_batch_size=cfg.export.max_batch_size,
            instance_kind=cfg.export.instance_kind,
        )
        logger.info("Staged %s -> %s", model_name, model_dir)

    missing = {
        DETECTOR_NAME, DENOISER_NAME, "resnet50_extractor", "vgg16_extractor"
    } - set(exported)
    if missing and only is None:
        logger.warning(
            "Model repository is incomplete: %s missing. Triton will fail to "
            "start until every directory has a 1/model.onnx.",
            ", ".join(sorted(missing)),
        )

    return {name: repo_dir / name for name in exported}
