"""Build the paired clean/noisy dataset CycleGAN trains on.

    clean signature  ──>  trainA/  (domain A)
           │
           └─ + form rules + caption text + stamp  ──>  trainB/  (domain B)

The generator learned in the B->A direction (`latest_net_G_B`) is the denoiser
the inference service runs.

Two behavioural fixes over the original dataset_preparation.py:

  * All randomness derives from one seed, threaded explicitly. The old code
    re-seeded a fixed literal inside the per-image line function, so every
    noisy image got an identical pair of rules.
  * Per-image failures are counted and reported, and the run fails if nothing
    was written. Previously a broad `except Exception` printed one line per
    skipped file and the summary still announced the full pair count, so an
    empty dataset looked like a successful build.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import Config
from .noise.document import DocumentNoise, seeded_rngs
from .noise.stamps import StampAugmentor

logger = logging.getLogger(__name__)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_images(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in VALID_EXTS)


def make_square(img: Image.Image, target: int) -> Image.Image:
    """Centre the signature on a white square, preserving aspect ratio."""
    img = img.convert("RGB")
    w, h = img.size
    size = max(target, w, h)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas.resize((target, target), Image.Resampling.LANCZOS)


def build(cfg: Config) -> dict[str, int]:
    """Write trainA/trainB/testA/testB under paths.cyclegan_dataset."""
    src = cfg.paths.resolve("raw_signatures")
    dst = cfg.paths.resolve("cyclegan_dataset")
    stamps_dir = cfg.paths.resolve("stamps")
    data_cfg = cfg.cyclegan_data

    paths = collect_images(src)
    if not paths:
        raise FileNotFoundError(
            f"No images under {src}. Run `sigtrain setup` for the expected "
            f"layout, or set paths.raw_signatures in configs/default.yaml."
        )
    logger.info("Found %d source images under %s", len(paths), src)

    np_rng, py_rng = seeded_rngs(data_cfg.seed)

    document = DocumentNoise(
        font_path=cfg.paths.font,
        rng=np_rng,
        p_lines=data_cfg.p_lines,
        p_text=data_cfg.p_text,
    )
    stamper = StampAugmentor(
        stamp_folder=str(stamps_dir),
        p_apply=data_cfg.p_stamp,
        rng=py_rng,
    )
    if not stamper._stamps:
        logger.warning(
            "No stamp images in %s - stamp noise disabled. The denoiser will "
            "not learn to remove seals.",
            stamps_dir,
        )

    dirs = {
        split + domain: dst / (split + domain)
        for split in ("train", "test")
        for domain in ("A", "B")
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    order = list(paths)
    py_rng.shuffle(order)
    n_test = max(1, int(len(order) * data_cfg.test_ratio))
    splits = {"test": order[:n_test], "train": order[n_test:]}

    written = {"train": 0, "test": 0}
    failures: list[tuple[Path, str]] = []

    for split, items in splits.items():
        for path in tqdm(items, desc=f"  {split}", unit="img"):
            try:
                clean, noisy = _make_pair(path, document, stamper, data_cfg.image_size)
            except Exception as exc:
                failures.append((path, str(exc)))
                continue
            stem = path.stem
            cv2.imwrite(str(dirs[f"{split}A"] / f"{stem}.png"), clean)
            cv2.imwrite(str(dirs[f"{split}B"] / f"{stem}.png"), noisy)
            written[split] += 1

    if failures:
        logger.warning("%d image(s) failed:", len(failures))
        for path, err in failures[:10]:
            logger.warning("  %s: %s", path, err)
        if len(failures) > 10:
            logger.warning("  ... and %d more", len(failures) - 10)

    total = written["train"] + written["test"]
    if total == 0:
        raise RuntimeError(
            f"Wrote 0 pairs from {len(paths)} source images - every image failed. "
            "See the errors above."
        )

    logger.info(
        "Wrote %d train pairs and %d test pairs to %s (%d failed)",
        written["train"],
        written["test"],
        dst,
        len(failures),
    )
    return {**written, "failed": len(failures)}


def _make_pair(
    path: Path,
    document: DocumentNoise,
    stamper: StampAugmentor,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(clean, noisy) as BGR uint8 arrays of shape (size, size, 3)."""
    square = make_square(Image.open(path), size)
    clean = cv2.cvtColor(np.array(square), cv2.COLOR_RGB2BGR)
    noisy = stamper(document(clean.copy()))
    return clean, noisy


def build_verification_split(cfg: Config) -> dict[str, int]:
    """Copy only genuine (non-`_forg`) person folders into the verification
    dataset, preserving the train/test split.

    Kept separate from the CycleGAN builder because the two want opposite
    things: CycleGAN needs clean images to corrupt, verification needs
    per-person folders to classify.
    """
    import shutil

    src = cfg.paths.resolve("raw_signatures")
    dst = cfg.paths.resolve("verification_dataset")
    counts = {}

    for split in ("train", "test"):
        src_split, dst_split = src / split, dst / split
        if not src_split.is_dir():
            logger.warning("Missing %s - skipping this split", src_split)
            counts[split] = 0
            continue
        dst_split.mkdir(parents=True, exist_ok=True)

        copied = existing = 0
        for folder in sorted(p for p in src_split.iterdir() if p.is_dir()):
            if "forg" in folder.name.lower():
                continue
            target = dst_split / folder.name
            if target.exists():
                existing += 1
            else:
                shutil.copytree(folder, target)
                copied += 1
        # Count what is PRESENT, not what this run copied. Counting only copies
        # made every re-run look like an empty dataset and abort the stage — so
        # `sigtrain all` could never be resumed once the split existed.
        counts[split] = copied + existing
        logger.info(
            "%s: %d genuine folder(s) (%d copied, %d already present)",
            split,
            counts[split],
            copied,
            existing,
        )

    if not any(counts.values()):
        raise RuntimeError(
            f"No genuine person folders found under {src}. Expected "
            f"{src}/train/<person>/ directories not ending in '_forg'."
        )
    return counts
