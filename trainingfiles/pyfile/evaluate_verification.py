"""
evaluate_verification.py
─────────────────────────────────────────────────────────────────────────────
Evaluates how well the feature extractor separates identities in embedding
space — no forgery data needed.

Two pair types built from test/_org folders only:
  Genuine pair   (label=1) : two signatures from the SAME person
  Impostor pair  (label=0) : one signature each from TWO DIFFERENT persons

A good extractor should produce:
  genuine  cosine similarity → close to 1.0  (same person clusters tightly)
  impostor cosine similarity → close to 0.0  (different persons pushed apart)

Preprocessing
─────────────
Uses backbone-specific Caffe mean subtraction via keras preprocess_input,
NOT simple /255 rescaling. This matches how the models were trained.

Usage
─────
    python evaluate_verification.py \
        --test_dir   ../data/cyclegan_unprocessed_data/test \
        --vgg16      ../model/verification_model/vgg16_extractor.keras \
        --resnet50   ../model/verification_model/resnet50_extractor.keras \
        --output_dir ../model/evaluation
"""

import argparse
import os
import warnings
from itertools import combinations
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.special import ndtri
from sklearn.metrics import auc, roc_curve
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.preprocessing import image as keras_image


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing  (Caffe-style, NOT /255)
# ─────────────────────────────────────────────────────────────────────────────

def load_vec(extractor, path: str, img_size: int, preprocess_fn) -> np.ndarray:
    """Load one image → Caffe preprocess → extract → L2-normalise."""
    img = keras_image.load_img(path, target_size=(img_size, img_size))
    arr = keras_image.img_to_array(img)          # float32  [0, 255]
    arr = preprocess_fn(np.expand_dims(arr, 0))  # subtract BGR means
    vec = extractor.predict(arr, verbose=0).flatten()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def get_person_dirs(test_dir: str) -> list:
    """
    Return genuine-only person folders from test_dir.

    Handles multiple naming conventions:
      001_org/  002_org/  ...   (Kaggle with suffix)
      001/      002/      ...   (plain numbered, _forg already removed)
      person001/ person002/ ... (any other naming)

    Always skips folders ending with '_forg'.
    """
    all_dirs = sorted([
        os.path.join(test_dir, d)
        for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
        and not d.endswith("_forg")
    ])

    if not all_dirs:
        raise FileNotFoundError(
            f"No subfolders found in '{test_dir}'. "
            "Check --test_dir points to the folder containing person subfolders."
        )

    # If _org folders exist use them; otherwise use all non-_forg folders
    org_dirs = [d for d in all_dirs if Path(d).name.endswith("_org")]
    chosen   = org_dirs if org_dirs else all_dirs

    print(f"  Found {len(chosen)} person folders "
          f"({'_org suffix' if org_dirs else 'plain folders'})")
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Pair building  (genuine _org only, no _forg)
# ─────────────────────────────────────────────────────────────────────────────

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def get_images(folder: str):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in VALID_EXTS
    ])


def build_pairs(test_dir: str, extractor, img_size: int, preprocess_fn):
    """
    Returns (scores, labels):
        label = 1  → same person    (genuine pair)
        label = 0  → different person (impostor pair)
    """
    org_dirs = get_person_dirs(test_dir)

    print(f"  Extracting embeddings …")
    person_vecs = {}
    for d in org_dirs:
        imgs = get_images(d)
        if not imgs:
            continue
        pid  = Path(d).name.replace("_org", "")
        vecs = [load_vec(extractor, p, img_size, preprocess_fn) for p in imgs]
        person_vecs[pid] = vecs

    scores, labels = [], []
    pids = list(person_vecs.keys())

    # ── Genuine pairs: same person ────────────────────────────────────────
    for pid, vecs in person_vecs.items():
        for a, b in combinations(vecs, 2):
            scores.append(float(np.dot(a, b)))
            labels.append(1)

    # ── Impostor pairs: different persons ─────────────────────────────────
    rng = np.random.default_rng(42)
    for p1, p2 in combinations(pids, 2):
        v1 = person_vecs[p1][rng.integers(len(person_vecs[p1]))]
        v2 = person_vecs[p2][rng.integers(len(person_vecs[p2]))]
        scores.append(float(np.dot(v1, v2)))
        labels.append(0)

    print(f"  Genuine pairs: {labels.count(1)}   "
          f"Impostor pairs: {labels.count(0)}")

    if len(set(labels)) < 2:
        raise ValueError(
            "Need both genuine AND impostor pairs for evaluation. "
            f"Got only labels: {set(labels)}. "
            "Ensure test_dir has at least 2 person subfolders with 2+ images each."
        )

    return np.array(scores), np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(fpr, tpr, thresholds):
    """
    Equal Error Rate — the threshold where FAR == FRR.
    Primary biometric metric: threshold-independent single number.
    Lower is better. < 0.10 is good, < 0.05 is strong.
    """
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


def compute_tar_at_far(fpr, tpr, thresholds,
                       far_targets=(0.05, 0.01, 0.001)):
    """
    TAR @ fixed FAR thresholds — operational deployment metric.
    "At a 1% false acceptance rate, what fraction of genuine pairs
    are correctly accepted?"
    Pick the FAR that matches your security requirement.
    """
    out = {}
    for target in far_targets:
        idx = np.where(fpr <= target)[0]
        if len(idx) == 0:
            out[target] = (0.0, float("nan"))
        else:
            best = idx[np.argmax(tpr[idx])]
            out[target] = (float(tpr[best]), float(thresholds[best]))
    return out


def compute_dprime(genuine_scores, impostor_scores):
    """
    d-prime (d') from signal detection theory.
    Measures how many standard deviations apart the two score
    distributions are, pooling their variances.
    d' > 1.0  reasonable
    d' > 2.0  good
    d' > 3.0  excellent
    Does NOT depend on a chosen threshold — purely about distribution shape.
    """
    mu_g, mu_i   = np.mean(genuine_scores), np.mean(impostor_scores)
    sig_g, sig_i = np.std(genuine_scores),  np.std(impostor_scores)
    denom = np.sqrt(0.5 * (sig_g**2 + sig_i**2))
    return float((mu_g - mu_i) / denom) if denom > 0 else 0.0


def compute_fnmr_at_fmr(fpr, tpr, thresholds,
                         fmr_targets=(0.05, 0.01, 0.001)):
    """
    FNMR @ fixed FMR — ISO/IEC 19795 standard reporting format.
    (FNMR = False Non-Match Rate = 1 - TAR,  FMR = False Match Rate = FAR)
    Complements TAR@FAR with the miss-rate perspective.
    """
    out = {}
    fnr = 1 - tpr
    for target in fmr_targets:
        idx = np.where(fpr <= target)[0]
        if len(idx) == 0:
            out[target] = 1.0
        else:
            best = idx[np.argmax(tpr[idx])]
            out[target] = float(fnr[best])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_score_dist(genuine, impostor, name, out_dir):
    """
    Histogram overlay of genuine vs impostor cosine similarities.
    Why: most intuitive check — well-separated peaks = good extractor.
    Overlap region = error-prone zone regardless of threshold.
    """
    bins = np.linspace(
        min(genuine.min(), impostor.min()),
        max(genuine.max(), impostor.max()), 60
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(genuine,  bins=bins, alpha=0.65, color="#1D9E75",
            label=f"Same person  μ={genuine.mean():.3f}", density=True)
    ax.hist(impostor, bins=bins, alpha=0.65, color="#D85A30",
            label=f"Diff person  μ={impostor.mean():.3f}", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"{name} — Score Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out_dir, f"{name}_score_dist.png"))


def plot_roc(fpr, tpr, roc_auc, name, out_dir):
    """
    ROC curve — TAR vs FAR across all thresholds.
    Why: shows the complete tradeoff space. AUC=1 perfect, AUC=0.5 random.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#185FA5", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (TAR)")
    ax.set_title(f"{name} — ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out_dir, f"{name}_roc.png"))


def plot_det(fpr, tpr, name, out_dir):
    """
    DET curve — FMR vs FNMR on normal-deviate (probit) scale.
    Why: ISO/IEC 19795 biometric standard. Stretches the low-error region
    making small differences at FAR=0.1% clearly visible where ROC cannot.
    """
    fnr = 1 - tpr
    eps = 1e-6
    fpr_c = np.clip(fpr, eps, 1 - eps)
    fnr_c = np.clip(fnr, eps, 1 - eps)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ndtri(fpr_c), ndtri(fnr_c), color="#8B3FA8", lw=2)
    ticks     = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    tick_lbls = [f"{t*100:.1f}%" for t in ticks]
    tick_vals = [ndtri(t) for t in ticks]
    ax.set_xticks(tick_vals); ax.set_xticklabels(tick_lbls, fontsize=7)
    ax.set_yticks(tick_vals); ax.set_yticklabels(tick_lbls, fontsize=7)
    ax.set_xlabel("FMR  (False Match Rate)")
    ax.set_ylabel("FNMR  (False Non-Match Rate)")
    ax.set_title(f"{name} — DET Curve")
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out_dir, f"{name}_det.png"))


def plot_threshold(fpr, tpr, thresholds, name, out_dir):
    """
    FAR and FRR vs decision threshold.
    Why: directly shows which threshold to use in production.
    The crossing point is the EER. Choose left of crossing for
    tighter security (lower FAR), right for higher convenience (lower FRR).
    """
    fnr = 1 - tpr
    thr = thresholds
    # roc_curve returns len(thr) = len(fpr)-1 in some sklearn versions
    n   = min(len(thr), len(fpr), len(fnr))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thr[:n], fpr[:n], color="#D85A30", lw=2,
            label="FAR — False Acceptance Rate")
    ax.plot(thr[:n], fnr[:n], color="#185FA5", lw=2,
            label="FRR — False Rejection Rate")
    ax.set_xlabel("Cosine Similarity Threshold")
    ax.set_ylabel("Error Rate")
    ax.set_title(f"{name} — Threshold vs Error Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out_dir, f"{name}_threshold.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Single-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(extractor_path: str, test_dir: str,
             name: str, output_dir: str, img_size: int = 224) -> dict:

    print(f"\n{'='*60}\n  Evaluating: {name.upper()}\n{'='*60}")

    extractor    = tf.keras.models.load_model(extractor_path, compile=False)
    preprocess_fn = vgg16_preprocess if "vgg" in name.lower() else resnet_preprocess
    print(f"  Extractor output: {extractor.output_shape[-1]}-d")

    scores, labels = build_pairs(test_dir, extractor, img_size, preprocess_fn)
    genuine  = scores[labels == 1]
    impostor = scores[labels == 0]

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc              = auc(fpr, tpr)
    eer, eer_thr         = compute_eer(fpr, tpr, thresholds)
    tar                  = compute_tar_at_far(fpr, tpr, thresholds)
    fnmr                 = compute_fnmr_at_fmr(fpr, tpr, thresholds)
    dprime               = compute_dprime(genuine, impostor)

    # ── Console report ────────────────────────────────────────────────────
    sep = "─" * 54
    print(f"""
  {sep}
  Pairs     same-person={len(genuine)}   diff-person={len(impostor)}
  {sep}
  METRIC                VALUE        WHAT IT MEANS
  {sep}
  EER              {eer:>8.4f}        lower is better  (< 0.10 good)
  AUC-ROC          {roc_auc:>8.4f}        higher is better (> 0.95 good)
  d-prime          {dprime:>8.4f}        higher is better (> 2.0 good)
  EER threshold    {eer_thr:>8.4f}        use this for production
  {sep}
  TAR @ FAR 5%     {tar[0.05][0]:>8.4f}        acceptance rate at loose security
  TAR @ FAR 1%     {tar[0.01][0]:>8.4f}        acceptance rate at normal security
  TAR @ FAR 0.1%   {tar[0.001][0]:>8.4f}        acceptance rate at strict security
  {sep}
  FNMR @ FMR 5%    {fnmr[0.05]:>8.4f}        miss rate at loose security
  FNMR @ FMR 1%    {fnmr[0.01]:>8.4f}        miss rate at normal security
  FNMR @ FMR 0.1%  {fnmr[0.001]:>8.4f}        miss rate at strict security
  {sep}
  Same-person   scores   μ={genuine.mean():.4f}   σ={genuine.std():.4f}
  Diff-person   scores   μ={impostor.mean():.4f}   σ={impostor.std():.4f}
  Gap (μ diff)           {genuine.mean() - impostor.mean():.4f}
  {sep}""")

    os.makedirs(output_dir, exist_ok=True)
    plot_score_dist(genuine, impostor, name, output_dir)
    plot_roc(fpr, tpr, roc_auc, name, output_dir)
    plot_det(fpr, tpr, name, output_dir)
    plot_threshold(fpr, tpr, thresholds, name, output_dir)

    return dict(name=name, eer=eer, eer_thr=eer_thr, auc=roc_auc,
                dprime=dprime, tar=tar, fnmr=fnmr,
                genuine_mean=float(genuine.mean()),
                impostor_mean=float(impostor.mean()),
                gap=float(genuine.mean() - impostor.mean()))


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results: list, output_dir: str):
    metrics = [
        ("EER ↓",     "eer",     "lower"),
        ("AUC-ROC ↑", "auc",     "higher"),
        ("d-prime ↑", "dprime",  "higher"),
        ("Score gap ↑","gap",    "higher"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    colors = ["#185FA5", "#D85A30"]
    names  = [r["name"] for r in results]

    for ax, (title, key, _) in zip(axes, metrics):
        vals = [r[key] for r in results]
        bars = ax.bar(names, vals, color=colors[:len(names)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.30)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("VGG16 vs ResNet50 — Embedding Quality", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [plot] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir",
                   required=True,
                   help="Test folder — only _org subfolders are used")
    p.add_argument("--vgg16",
                   default="../model/verification_model/vgg16_extractor.keras")
    p.add_argument("--resnet50",
                   default="../model/verification_model/resnet50_extractor.keras")
    p.add_argument("--output_dir", default="../model/evaluation")
    p.add_argument("--img_size",   type=int, default=224)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for path, name in [(args.vgg16, "vgg16"), (args.resnet50, "resnet50")]:
        if os.path.isfile(path):
            results.append(evaluate(path, args.test_dir, name,
                                    args.output_dir, args.img_size))
        else:
            print(f"[skip] not found: {path}")

    if len(results) == 2:
        plot_comparison(results, args.output_dir)

    print("\n[done]  outputs →", args.output_dir)