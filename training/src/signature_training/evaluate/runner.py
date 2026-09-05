"""Evaluate the trained feature extractors on held-out identities.

The people in test/ never appear in train/, so this measures what actually
matters: whether the embedding separates someone the model has never seen.

The number to carry forward is the EER threshold. It is a cosine *similarity*;
`inference/.env` wants a cosine *distance*, so set

    MATCH_THRESHOLD = 1 - eer_threshold
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve

from ..config import Config
from . import metrics, pairs, plots

logger = logging.getLogger(__name__)

BACKBONES = ("vgg16", "resnet50")


def _embedder(extractor, backbone: str, size: int):
    """Image path -> L2-normalised embedding, matching training preprocessing."""
    import numpy as np
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pp
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pp
    from tensorflow.keras.preprocessing import image as keras_image

    preprocess = vgg_pp if backbone == "vgg16" else resnet_pp

    def embed(path: Path) -> np.ndarray:
        img = keras_image.load_img(str(path), target_size=(size, size))
        arr = preprocess(np.expand_dims(keras_image.img_to_array(img), 0))
        vec = extractor.predict(arr, verbose=0).flatten()
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    return embed


def evaluate_one(cfg: Config, backbone: str, extractor_path: Path, test_dir: Path) -> dict:
    import tensorflow as tf

    logger.info("=== evaluating %s ===", backbone)
    extractor = tf.keras.models.load_model(extractor_path, compile=False)
    logger.info("Extractor output: %d-d", extractor.output_shape[-1])

    scores, labels = pairs.build(
        test_dir,
        _embedder(extractor, backbone, cfg.evaluate.image_size),
        impostor_pairs_per_couple=cfg.evaluate.impostor_pairs_per_couple,
        seed=cfg.evaluate.seed,
    )
    genuine, impostor = scores[labels == 1], scores[labels == 0]

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_threshold = metrics.equal_error_rate(fpr, tpr, thresholds)
    tar = metrics.tar_at_far(fpr, tpr, thresholds)
    fnmr = metrics.fnmr_at_fmr(fpr, tpr)
    dprime = metrics.d_prime(genuine, impostor)
    floor = metrics.resolvable_far(len(impostor))

    out_dir = cfg.paths.resolve("evaluation")
    plots.score_distribution(genuine, impostor, backbone, out_dir)
    plots.roc(fpr, tpr, roc_auc, backbone, out_dir)
    plots.det(fpr, tpr, backbone, out_dir)
    plots.threshold_sweep(fpr, tpr, thresholds, backbone, out_dir)

    result = {
        "name": backbone,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "match_threshold_for_service": 1.0 - eer_threshold,
        "auc": float(roc_auc),
        "dprime": dprime,
        "genuine_pairs": int(len(genuine)),
        "impostor_pairs": int(len(impostor)),
        "resolvable_far": floor,
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "gap": float(genuine.mean() - impostor.mean()),
        "tar_at_far": {str(k): v[0] for k, v in tar.items()},
        "fnmr_at_fmr": {str(k): v for k, v in fnmr.items()},
    }
    _report(result, tar, fnmr, genuine, impostor)
    return result


def _report(r: dict, tar: dict, fnmr: dict, genuine: np.ndarray, impostor: np.ndarray) -> None:
    sep = "─" * 58
    lines = [
        "",
        f"  {sep}",
        f"  {r['name'].upper()}   genuine={r['genuine_pairs']}  impostor={r['impostor_pairs']}",
        f"  {sep}",
        f"  EER              {r['eer']:>8.4f}     lower is better  (< 0.10 good)",
        f"  AUC-ROC          {r['auc']:>8.4f}     higher is better (> 0.95 good)",
        f"  d-prime          {r['dprime']:>8.4f}     higher is better (> 2.0 good)",
        f"  {sep}",
        f"  EER threshold    {r['eer_threshold']:>8.4f}     cosine SIMILARITY",
        f"  -> set inference MATCH_THRESHOLD={r['match_threshold_for_service']:.4f}  (cosine DISTANCE)",
        f"  {sep}",
    ]
    for target in (0.05, 0.01, 0.001):
        note = "" if target >= r["resolvable_far"] else "  << below sample resolution"
        lines.append(f"  TAR @ FAR {target*100:>5.1f}%  {tar[target][0]:>8.4f}{note}")
    lines.append(f"  {sep}")
    for target in (0.05, 0.01, 0.001):
        note = "" if target >= r["resolvable_far"] else "  << below sample resolution"
        lines.append(f"  FNMR @ FMR {target*100:>4.1f}% {fnmr[target]:>8.4f}{note}")
    lines += [
        f"  {sep}",
        f"  Same person   μ={genuine.mean():.4f}  σ={genuine.std():.4f}",
        f"  Diff person   μ={impostor.mean():.4f}  σ={impostor.std():.4f}",
        f"  Gap                    {r['gap']:.4f}",
        f"  {sep}",
    ]
    if r["resolvable_far"] > 0.001:
        lines.append(
            f"  NOTE: {r['impostor_pairs']} impostor pairs resolve FAR only to "
            f"{r['resolvable_far']*100:.3f}%. Raise "
            f"evaluate.impostor_pairs_per_couple for finer low-FAR readings."
        )
        lines.append(f"  {sep}")
    print("\n".join(lines))


def run(cfg: Config) -> list[dict]:
    models_dir = cfg.paths.resolve("models")
    test_dir = cfg.paths.resolve("verification_dataset") / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"{test_dir} not found. Run `sigtrain data-verification` first."
        )

    out_dir = cfg.paths.resolve("evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for backbone in BACKBONES:
        path = models_dir / f"{backbone}_extractor.keras"
        if not path.is_file():
            logger.info("Skipping %s - %s not found", backbone, path.name)
            continue
        results.append(evaluate_one(cfg, backbone, path, test_dir))

    if not results:
        raise FileNotFoundError(
            f"No extractors found in {models_dir}. Run `sigtrain train-verification`."
        )

    if len(results) > 1:
        plots.comparison(results, out_dir)

    summary = out_dir / "metrics.json"
    summary.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", summary)
    return results
