"""Build genuine and impostor pairs from held-out person folders.

    genuine  (label 1) : two signatures from the same person
    impostor (label 0) : one signature each from two different people

No forgeries are involved. This measures whether the embedding separates
*identities*, which is what the deployed nearest-neighbour search relies on.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def person_dirs(test_dir: Path) -> list[Path]:
    """Genuine-only person folders.

    Tolerates the naming conventions this dataset appears under: `001_org/`
    (Kaggle), plain `001/` once forgeries have been filtered out, or anything
    else. Folders ending in `_forg` are always excluded.
    """
    all_dirs = sorted(
        d for d in test_dir.iterdir() if d.is_dir() and not d.name.endswith("_forg")
    )
    if not all_dirs:
        raise FileNotFoundError(
            f"No person subfolders in {test_dir}. paths.verification_dataset "
            "must point at the folder whose train/ and test/ *contain* the "
            "per-person directories. Run `sigtrain data-verification` first."
        )
    org = [d for d in all_dirs if d.name.endswith("_org")]
    chosen = org or all_dirs
    logger.info("Found %d person folder(s)", len(chosen))
    return chosen


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS)


def build(
    test_dir: Path,
    embed_fn,
    impostor_pairs_per_couple: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """(scores, labels) as cosine similarities between L2-normalised vectors.

    `impostor_pairs_per_couple` controls how finely the low-FAR region of the
    ROC can be resolved. The original code sampled exactly one pair per couple,
    giving C(n, 2) impostor scores — 210 for a 21-person split — so the FPR grid
    stepped by 0.48% and the reported "TAR @ FAR 0.1%" was really TAR at FAR 0.
    """
    rng = np.random.default_rng(seed)

    person_vecs: dict[str, list[np.ndarray]] = {}
    for folder in person_dirs(test_dir):
        paths = images(folder)
        if len(paths) < 1:
            continue
        person_vecs[folder.name.replace("_org", "")] = [embed_fn(p) for p in paths]

    scores: list[float] = []
    labels: list[int] = []

    # Genuine: every within-person pair.
    for vecs in person_vecs.values():
        for a, b in combinations(vecs, 2):
            scores.append(float(np.dot(a, b)))
            labels.append(1)

    # Impostor: up to k random cross-person pairs per couple.
    pids = list(person_vecs)
    for p1, p2 in combinations(pids, 2):
        v1s, v2s = person_vecs[p1], person_vecs[p2]
        k = min(impostor_pairs_per_couple, len(v1s) * len(v2s))
        seen = set()
        for _ in range(k):
            i, j = int(rng.integers(len(v1s))), int(rng.integers(len(v2s)))
            if (i, j) in seen:
                continue
            seen.add((i, j))
            scores.append(float(np.dot(v1s[i], v2s[j])))
            labels.append(0)

    n_gen, n_imp = labels.count(1), labels.count(0)
    logger.info("Genuine pairs: %d   Impostor pairs: %d", n_gen, n_imp)
    if n_gen == 0 or n_imp == 0:
        raise ValueError(
            f"Need both genuine and impostor pairs; got {n_gen} and {n_imp}. "
            "The test directory needs at least 2 people with 2+ images each."
        )

    return np.asarray(scores), np.asarray(labels)
