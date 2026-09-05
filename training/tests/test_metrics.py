"""Verification metrics."""

import numpy as np
import pytest
from sklearn.metrics import roc_curve

from signature_training.evaluate import metrics


def _curve(genuine, impostor):
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    return roc_curve(labels, scores)


def test_perfect_separation_gives_zero_eer():
    fpr, tpr, thr = _curve(np.full(50, 0.9), np.full(50, 0.1))
    eer, _ = metrics.equal_error_rate(fpr, tpr, thr)
    assert eer == pytest.approx(0.0, abs=1e-9)


def test_random_scores_give_eer_near_a_half():
    rng = np.random.default_rng(0)
    fpr, tpr, thr = _curve(rng.random(500), rng.random(500))
    eer, _ = metrics.equal_error_rate(fpr, tpr, thr)
    assert 0.4 < eer < 0.6


def test_eer_threshold_sits_between_the_distributions():
    rng = np.random.default_rng(1)
    genuine = rng.normal(0.8, 0.05, 300)
    impostor = rng.normal(0.2, 0.05, 300)
    fpr, tpr, thr = _curve(genuine, impostor)
    _, threshold = metrics.equal_error_rate(fpr, tpr, thr)
    assert impostor.mean() < threshold < genuine.mean()


def test_d_prime_grows_with_separation():
    rng = np.random.default_rng(2)
    close = metrics.d_prime(rng.normal(0.5, 0.1, 500), rng.normal(0.4, 0.1, 500))
    far = metrics.d_prime(rng.normal(0.9, 0.1, 500), rng.normal(0.1, 0.1, 500))
    assert far > close > 0


def test_d_prime_handles_zero_variance():
    assert metrics.d_prime(np.full(10, 0.5), np.full(10, 0.5)) == 0.0


def test_tar_at_far_is_monotone_in_the_target():
    rng = np.random.default_rng(3)
    fpr, tpr, thr = _curve(rng.normal(0.8, 0.1, 400), rng.normal(0.3, 0.1, 400))
    tar = metrics.tar_at_far(fpr, tpr, thr)
    assert tar[0.05][0] >= tar[0.01][0] >= tar[0.001][0]


def test_fnmr_is_one_minus_tar():
    rng = np.random.default_rng(4)
    fpr, tpr, thr = _curve(rng.normal(0.8, 0.1, 400), rng.normal(0.3, 0.1, 400))
    tar = metrics.tar_at_far(fpr, tpr, thr)
    fnmr = metrics.fnmr_at_fmr(fpr, tpr)
    for target in (0.05, 0.01, 0.001):
        assert fnmr[target] == pytest.approx(1 - tar[target][0], abs=1e-9)


# ── the sampling-resolution guard ─────────────────────────────────────────────


def test_resolvable_far_reflects_the_impostor_count():
    """210 impostor pairs — what one-pair-per-couple gives on the 21-person
    test split — cannot express a 0.1% FAR at all."""
    assert metrics.resolvable_far(210) == pytest.approx(1 / 210)
    assert metrics.resolvable_far(210) > 0.001      # 0.1% unresolvable
    assert metrics.resolvable_far(5000) < 0.001     # resolvable
    assert metrics.resolvable_far(0) == 1.0


def test_far_below_resolution_collapses_onto_far_zero():
    """Demonstrates why the guard exists rather than asserting an opinion."""
    rng = np.random.default_rng(5)
    fpr, tpr, thr = _curve(rng.normal(0.8, 0.1, 400), rng.normal(0.3, 0.1, 210))
    tar = metrics.tar_at_far(fpr, tpr, thr, targets=(0.001,))
    at_zero = metrics.tar_at_far(fpr, tpr, thr, targets=(0.0,))
    assert tar[0.001][0] == at_zero[0.0][0]
