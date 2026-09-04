"""Evaluation plots.

Four views of the same scores, each answering a different question:

  score distribution  are the two populations separated at all?
  ROC                 the whole TAR/FAR trade-off space
  DET                 the same on a probit scale, which stretches the low-error
                      corner where the operating point actually lives
  threshold sweep     which threshold to configure, read directly off the axis
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a training container
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import ndtri  # noqa: E402

logger = logging.getLogger(__name__)

GENUINE_C, IMPOSTOR_C, ROC_C, DET_C = "#1D9E75", "#D85A30", "#185FA5", "#8B3FA8"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  plot %s", path)


def score_distribution(genuine, impostor, name: str, out_dir: Path) -> None:
    bins = np.linspace(
        min(genuine.min(), impostor.min()), max(genuine.max(), impostor.max()), 60
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(genuine, bins=bins, alpha=0.65, color=GENUINE_C, density=True,
            label=f"Same person  μ={genuine.mean():.3f}")
    ax.hist(impostor, bins=bins, alpha=0.65, color=IMPOSTOR_C, density=True,
            label=f"Diff person  μ={impostor.mean():.3f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"{name} — score distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir / f"{name}_score_dist.png")


def roc(fpr, tpr, roc_auc: float, name: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=ROC_C, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False acceptance rate (FAR)")
    ax.set_ylabel("True acceptance rate (TAR)")
    ax.set_title(f"{name} — ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, out_dir / f"{name}_roc.png")


def det(fpr, tpr, name: str, out_dir: Path) -> None:
    eps = 1e-6
    fpr_c = np.clip(fpr, eps, 1 - eps)
    fnr_c = np.clip(1 - tpr, eps, 1 - eps)
    ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ndtri(fpr_c), ndtri(fnr_c), color=DET_C, lw=2)
    ax.set_xticks([ndtri(t) for t in ticks])
    ax.set_xticklabels([f"{t*100:.1f}%" for t in ticks], fontsize=7)
    ax.set_yticks([ndtri(t) for t in ticks])
    ax.set_yticklabels([f"{t*100:.1f}%" for t in ticks], fontsize=7)
    ax.set_xlabel("FMR (false match rate)")
    ax.set_ylabel("FNMR (false non-match rate)")
    ax.set_title(f"{name} — DET")
    ax.grid(alpha=0.3)
    _save(fig, out_dir / f"{name}_det.png")


def threshold_sweep(fpr, tpr, thresholds, name: str, out_dir: Path) -> None:
    fnr = 1 - tpr
    n = min(len(thresholds), len(fpr), len(fnr))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds[:n], fpr[:n], color=IMPOSTOR_C, lw=2, label="FAR — false acceptance")
    ax.plot(thresholds[:n], fnr[:n], color=ROC_C, lw=2, label="FRR — false rejection")
    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("Error rate")
    ax.set_title(f"{name} — threshold vs error (crossing = EER)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir / f"{name}_threshold.png")


def comparison(results: list[dict], out_dir: Path) -> None:
    """Side-by-side bars for the headline metrics of each backbone."""
    metrics = [("EER ↓", "eer"), ("AUC-ROC ↑", "auc"),
               ("d-prime ↑", "dprime"), ("Score gap ↑", "gap")]
    names = [r["name"] for r in results]
    colors = [ROC_C, IMPOSTOR_C]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, (title, key) in zip(axes, metrics):
        vals = [r[key] for r in results]
        bars = ax.bar(names, vals, color=colors[: len(names)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.30 if max(vals) > 0 else 1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Embedding quality by backbone", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir / "comparison.png")
