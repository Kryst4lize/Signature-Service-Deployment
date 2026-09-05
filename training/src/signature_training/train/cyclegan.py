"""Drive CycleGAN training in the upstream junyanz/pytorch-CycleGAN-and-pix2pix repo.

The repo is not vendored — it has its own licence and release cadence, and its
`train.py` is the supported entry point. `sigtrain setup` clones it; this module
invokes it with arguments derived from the pipeline config, so the dataset
location and the checkpoint location cannot drift apart the way they do when
the command is copy-pasted from a notes file.

Domain orientation, which everything downstream depends on:

    trainA = clean          trainB = noisy
    G_A : A -> B  (clean -> noisy)      unused at inference
    G_B : B -> A  (noisy -> clean)      <- this is the denoiser we deploy

so `latest_net_G_B.pth` is the checkpoint that becomes `latest_net_G_B.onnx`.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git"


def ensure_repo(cfg: Config) -> Path:
    repo = cfg.paths.resolve("cyclegan_repo")
    if (repo / "train.py").is_file():
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s", REPO_URL, repo)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
    return repo


def build_command(cfg: Config, repo: Path, resume_epoch: int | None = None) -> list[str]:
    t = cfg.cyclegan_train
    cmd = [
        sys.executable,
        "train.py",
        "--dataroot",
        str(cfg.paths.resolve("cyclegan_dataset")),
        "--checkpoints_dir",
        str(cfg.paths.resolve("cyclegan_checkpoints")),
        "--name",
        t.name,
        "--model",
        "cycle_gan",
        "--norm",
        t.norm,
        "--load_size",
        str(t.load_size),
        "--crop_size",
        str(t.crop_size),
        "--n_epochs",
        str(t.n_epochs),
        "--n_epochs_decay",
        str(t.n_epochs_decay),
        "--gpu_ids",
        t.gpu_ids,
    ]
    if resume_epoch is not None:
        cmd += [
            "--continue_train",
            "--epoch",
            str(resume_epoch),
            "--epoch_count",
            str(resume_epoch + 1),
        ]
    cmd += list(t.extra_args)
    return cmd


def run(cfg: Config, resume_epoch: int | None = None) -> Path:
    """Train, and return the directory holding latest_net_G_B.pth."""
    repo = ensure_repo(cfg)

    dataroot = cfg.paths.resolve("cyclegan_dataset")
    if not (dataroot / "trainA").is_dir():
        raise FileNotFoundError(f"{dataroot}/trainA not found. Run `sigtrain data-cyclegan` first.")

    cmd = build_command(cfg, repo, resume_epoch)
    logger.info("Running: %s", " ".join(cmd))
    logger.info("  (cwd=%s)", repo)
    subprocess.run(cmd, cwd=repo, check=True)

    out = cfg.paths.resolve("cyclegan_checkpoints") / cfg.cyclegan_train.name
    logger.info("Checkpoints in %s; the denoiser is latest_net_G_B.pth", out)
    return out
