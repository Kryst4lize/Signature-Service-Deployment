"""Biometric verification metrics.

All of these are threshold-independent or report a full operating curve,
because a single accuracy number says nothing useful about a system whose
decision threshold you still have to choose.
"""

from __future__ import annotations

import numpy as np


def equal_error_rate(fpr, tpr, thresholds) -> tuple[float, float]:
    """EER and the score at which it occurs.

    The point where the false-accept and false-reject rates cross. The single
    most useful number here: one value, no threshold to pick first.
    Below 0.10 is good, below 0.05 is strong.

    The returned threshold is a cosine *similarity*. The inference service
    compares cosine *distance*, so its MATCH_THRESHOLD is 1 - this value.
    """
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


def tar_at_far(fpr, tpr, thresholds, targets=(0.05, 0.01, 0.001)) -> dict:
    """True-accept rate at fixed false-accept rates.

    The operational question: "at a 1% false-accept rate, what fraction of
    genuine signatures do we accept?" Pick the FAR your security posture
    requires and read off the cost.

    A target finer than 1/n_impostor_pairs cannot be resolved and collapses onto
    the FAR = 0 point; `resolvable()` reports where that boundary is.
    """
    out = {}
    for target in targets:
        idx = np.where(fpr <= target)[0]
        if len(idx) == 0:
            out[target] = (0.0, float("nan"))
        else:
            best = idx[int(np.argmax(tpr[idx]))]
            out[target] = (float(tpr[best]), float(thresholds[best]))
    return out


def fnmr_at_fmr(fpr, tpr, targets=(0.05, 0.01, 0.001)) -> dict:
    """False non-match rate at fixed false-match rates (ISO/IEC 19795 form)."""
    out = {}
    fnr = 1 - tpr
    for target in targets:
        idx = np.where(fpr <= target)[0]
        out[target] = 1.0 if len(idx) == 0 else float(fnr[idx[int(np.argmax(tpr[idx]))]])
    return out


def d_prime(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Separation between the two score distributions in pooled standard
    deviations. Threshold-free — purely about distribution shape.
    > 1 reasonable, > 2 good, > 3 excellent.
    """
    denom = np.sqrt(0.5 * (genuine.std() ** 2 + impostor.std() ** 2))
    return float((genuine.mean() - impostor.mean()) / denom) if denom > 0 else 0.0


def resolvable_far(n_impostor_pairs: int) -> float:
    """Smallest FAR the impostor sample can express.

    With k impostor pairs the FPR grid has step 1/k, so a "TAR @ FAR 0.1%"
    computed from 210 pairs is really TAR at FAR = 0. Reporting a number that
    the sample cannot support is worse than reporting none.
    """
    return 1.0 / n_impostor_pairs if n_impostor_pairs else 1.0
