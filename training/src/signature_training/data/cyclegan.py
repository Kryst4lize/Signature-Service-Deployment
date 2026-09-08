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
from . import colour
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
                clean, noisy = _make_pair(
                    path, document, stamper, data_cfg.image_size, cfg.colour.mode
                )
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
    colour_mode: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """(clean, noisy) as BGR uint8 arrays of shape (size, size, 3).

    Colour correction is applied to the CLEAN image before noise is added, so
    domain A is what the generator learns to produce and domain B inherits the
    same paper colour. Correcting only one domain would teach the generator to
    change colour as part of denoising.
    """
    square = make_square(Image.open(path), size)
    clean = cv2.cvtColor(np.array(square), cv2.COLOR_RGB2BGR)
    if colour_mode != "none":
        clean = colour.apply(clean, colour_mode).astype(np.uint8)
    noisy = stamper(document(clean.copy()))
    return clean, noisy


COLOUR_STAMP = ".colour_mode"


def read_colour_stamp(dataset: Path) -> str:
    """The colour mode the dataset on disk was written under.

    An unstamped directory that already holds people is `none`, and that is a
    fact rather than an assumption: before the correction moved into this
    builder it was a bare `shutil.copytree`, so every dataset predating the
    stamp is provably uncorrected.

    An unstamped EMPTY directory has no content to describe, so it reports
    whatever the caller is about to write.
    """
    stamp = dataset / COLOUR_STAMP
    if stamp.is_file():
        return stamp.read_text().strip()
    return "none" if any(dataset.glob("*/*/")) else ""


def _check_colour_stamp(dataset: Path, mode: str) -> None:
    """Refuse to extend a dataset that was written under a different colour mode.

    `build_verification_split` skips person folders that already exist, so
    without this a run that changed `colour.mode` would leave the previously
    written folders uncorrected and report success. The result trains on a
    mixture of two colour distributions — the exact defect this setting exists
    to remove, made invisible by an incremental build.

    The unstamped case is the one that matters, and an earlier version of this
    function got it backwards. It assumed an unstamped dataset matched whatever
    the current run wanted, which is wrong in precisely the situation the guard
    exists for: the first time anyone sets `colour.mode: whiten`, every existing
    dataset is unstamped and uncorrected. The guard would skip every folder,
    write a stamp asserting `whiten`, log "assuming colour.mode='whiten'" as
    though it had checked, and return full counts — training then ran on a
    +8.18 corpus certified as corrected, and the one command that would have
    fixed it (`mode=none`) was now refused by the false stamp.

    Read-only. Recording the mode is `_write_colour_stamp`, and it happens after
    a build succeeds rather than before it starts: creating the directory and
    stamping it up front meant a run that failed on a mistyped `raw_signatures`
    left behind an empty stamped directory, which then refused every later mode
    change — claiming to hold images written under a mode, having written none.
    """
    previous = read_colour_stamp(dataset)
    if previous and previous != mode:
        raise RuntimeError(
            f"{dataset} holds images written with colour.mode={previous!r}, but this "
            f"run has colour.mode={mode!r}. Existing person folders are skipped, so "
            f"continuing would mix two colour distributions in one dataset.\n"
            f"Delete {dataset} and re-run `sigtrain data-verification`."
            + (
                f"\n(There is no {COLOUR_STAMP} file. The directory predates it, and "
                f"the builder did not correct colour at all back then, so its contents "
                f"are necessarily {previous!r}.)"
                if not (dataset / COLOUR_STAMP).is_file()
                else ""
            )
        )


def _write_colour_stamp(dataset: Path, mode: str) -> None:
    """Record the mode, once there is something for it to describe."""
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / COLOUR_STAMP).write_text(f"{mode}\n")


def require_colour_stamp(dataset: Path, mode: str) -> None:
    """Assert that a dataset about to be READ carries the configured mode.

    The builder's guard alone left a gap: nothing outside this module consulted
    the stamp, so `sigtrain train-verification` would happily train on a
    corrected corpus with the correction configured off, or the reverse, and say
    nothing. Both produce a model whose colour distribution is not the one
    recorded next to it.
    """
    previous = read_colour_stamp(dataset)
    if previous and previous != mode:
        raise RuntimeError(
            f"{dataset} was built with colour.mode={previous!r}, but this run has "
            f"colour.mode={mode!r}. The images on disk carry the correction; "
            f"re-running with a mismatched setting trains on one distribution while "
            f"recording another.\n"
            f"Either set colour.mode={previous!r}, or delete {dataset} and re-run "
            f"`sigtrain data-verification`."
        )


def _copy_person(folder: Path, target: Path, colour_mode: str) -> None:
    """Materialise one person's folder in the verification dataset.

    A plain copy when no colour correction is configured, and a re-encode when
    there is. Correcting HERE rather than only in the Keras
    `preprocessing_function` is deliberate: Keras runs that hook AFTER
    augmentation, so it sees the borders rotation and shift fill in with
    `cval=255`. Those neutral pixels sit inside the paper band and pull the
    per-channel means together, weakening the very correction being applied — at
    the mean 10.8% fill that augmentation produces, the dataset cast comes out at
    +0.83 instead of +0.00. At build time there is no augmentation and the
    estimate is clean.
    """
    import shutil

    if colour_mode == "none":
        shutil.copytree(folder, target)
        return

    target.mkdir(parents=True, exist_ok=True)
    for image in sorted(p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS):
        src = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if src is None:  # not decodable as an image; copy it through untouched
            shutil.copy2(image, target / image.name)
            continue
        cv2.imwrite(str(target / image.name), colour.apply(src, colour_mode).astype(np.uint8))


def build_verification_split(cfg: Config) -> dict[str, int]:
    """Copy only genuine (non-`_forg`) person folders into the verification
    dataset, preserving the train/test split.

    Kept separate from the CycleGAN builder because the two want opposite
    things: CycleGAN needs clean images to corrupt, verification needs
    per-person folders to classify.

    `colour.mode` is applied on the way in, so what is on disk is exactly what
    the backbones train on.
    """
    src = cfg.paths.resolve("raw_signatures")
    dst = cfg.paths.resolve("verification_dataset")
    counts = {}
    _check_colour_stamp(dst, cfg.colour.mode)
    if cfg.colour.mode != "none":
        logger.info("Applying colour mode %r while writing %s", cfg.colour.mode, dst)

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
                _copy_person(folder, target, cfg.colour.mode)
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
    # Only now, with images actually on disk for it to describe.
    _write_colour_stamp(dst, cfg.colour.mode)
    return counts
