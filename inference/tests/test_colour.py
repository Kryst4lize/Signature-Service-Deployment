"""Paper white-balance, and the contract between the two copies of it.

`inference/api/app/colour.py` is a mirror of
`training/src/signature_training/data/colour.py`. They are separate packages
with no shared import — the same arrangement as the Caffe preprocessing — so
the last test here compares them line by line.
"""

import numpy as np
import pytest

from app.colour import apply, desaturate, estimate_paper, whiten

# Measured on 400 genuine images of the Kaggle training set (paper = luminance
# > 200): red sits ~8 levels below green and blue, which reads as a cyan cast.
PAPER_CAST = np.array([243.28, 251.47, 251.46], dtype=np.float32)


def _page(paper=PAPER_CAST, ink=(102.0, 90.0, 96.0), size=128, ink_frac=0.15):
    """A synthetic scan: mostly paper, a minority of ink."""
    img = np.tile(np.asarray(paper, dtype=np.float32), (size, size, 1))
    rng = np.random.default_rng(0)
    img += rng.normal(scale=1.0, size=img.shape).astype(np.float32)
    n = int(size * size * ink_frac)
    ys = rng.integers(0, size, n)
    xs = rng.integers(0, size, n)
    img[ys, xs] = np.asarray(ink, dtype=np.float32)
    return np.clip(img, 0, 255)


def _paper_cast(img):
    lum = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    bg = img[lum > 200]
    m = bg.mean(axis=0)
    return float(m[2] - m[0])


# ── the estimator ─────────────────────────────────────────────────────────────


def test_estimate_paper_finds_the_paper_not_the_ink():
    got = estimate_paper(_page())
    np.testing.assert_allclose(got, PAPER_CAST, atol=1.5)


def test_estimate_paper_survives_a_mostly_ink_crop():
    """A tight crop can be more ink than paper; the estimate must not collapse
    onto the ink colour."""
    got = estimate_paper(_page(ink_frac=0.75))
    assert got.min() > 200, got


def test_estimate_paper_is_not_fooled_by_a_saturated_spike():
    """The failure that made the first two implementations no-ops.

    Real denoiser output has paper spread from ~249 up to a spike at 255. A
    percentile or a top-quantile median lands on the spike and reports ~255 for
    every channel, so the computed gain is 1.0 and nothing happens.
    """
    img = _page(paper=(249.9, 253.4, 253.4), ink_frac=0.05)
    img[:20] = 255.0  # the saturated spike
    got = estimate_paper(img)
    assert got[0] < 253.0, f"estimator saturated: {got}"
    assert got[1] - got[0] > 1.5, f"cast invisible to the estimator: {got}"


# ── the correction ────────────────────────────────────────────────────────────


def test_whiten_removes_the_measured_dataset_cast():
    before = _page()
    assert _paper_cast(before) > 7.0
    after = _paper_cast(whiten(before))
    assert abs(after) < 1.0, f"cast still {after:+.2f}"


def test_whiten_equalises_rather_than_clipping_to_white():
    """Targeting `white` looks equivalent and is not: when paper is already near
    saturation the extra gain is swallowed by the clip and the cast survives."""
    img = _page(paper=(249.9, 253.4, 253.4), ink_frac=0.05)
    assert abs(_paper_cast(whiten(img))) < abs(_paper_cast(img))


def test_whiten_only_brightens():
    img = _page()
    assert whiten(img).min() >= img.min() - 1e-3


def test_whiten_preserves_stroke_geometry():
    """Correction must not erode strokes: ink stays far darker than paper."""
    img = _page()
    out = whiten(img)
    lum_in = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum_out = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    ink = lum_in < 150
    assert ink.sum() > 100
    # Ink stays dark, and separation from paper does not shrink.
    assert lum_out[ink].mean() < 160
    assert (lum_out[~ink].mean() - lum_out[ink].mean()) >= (
        lum_in[~ink].mean() - lum_in[ink].mean()
    ) - 1.0


def test_strength_scales_the_correction():
    img = _page()
    full = abs(_paper_cast(whiten(img, strength=1.0)))
    half = abs(_paper_cast(whiten(img, strength=0.5)))
    none = abs(_paper_cast(whiten(img, strength=0.0)))
    assert full < half < none


def test_max_gain_caps_a_dark_image():
    """A dark photograph is not a tinted scan; do not stretch it to white."""
    dark = _page(paper=(60.0, 62.0, 62.0), ink_frac=0.05)
    out = whiten(dark, max_gain=1.6)
    assert out.max() <= 255.0
    assert out[..., 1].mean() / max(dark[..., 1].mean(), 1e-6) <= 1.61


def test_whiten_works_in_unit_range():
    img = _page() / 255.0
    out = whiten(img, white=1.0)
    assert out.max() <= 1.0
    assert abs(_paper_cast(out * 255.0)) < 1.0


# ── desaturate ────────────────────────────────────────────────────────────────


def test_desaturate_removes_all_chroma():
    out = desaturate(_page())
    np.testing.assert_allclose(out[..., 0], out[..., 1], atol=1e-4)
    np.testing.assert_allclose(out[..., 1], out[..., 2], atol=1e-4)


def test_apply_dispatch():
    img = _page()
    np.testing.assert_allclose(apply(img, "none"), img)
    assert abs(_paper_cast(apply(img, "whiten"))) < 1.0
    assert abs(_paper_cast(apply(img, "desaturate"))) < 1e-4
    with pytest.raises(ValueError, match="Unknown colour mode"):
        apply(img, "sepia")


# ── the two copies must not drift ─────────────────────────────────────────────


def test_inference_copy_matches_the_training_copy():
    """Byte-identical apart from the module docstring, which names the other side."""
    import pathlib

    here = pathlib.Path(__file__).resolve().parents[1] / "api" / "app" / "colour.py"
    there = (
        pathlib.Path(__file__).resolve().parents[2]
        / "training"
        / "src"
        / "signature_training"
        / "data"
        / "colour.py"
    )
    if not there.is_file():  # inference checked out on its own
        pytest.skip("training half not present")

    def body(p):
        text = p.read_text()
        return text[text.index("from __future__") :]

    assert body(here) == body(there), (
        "inference/api/app/colour.py has drifted from "
        "training/src/signature_training/data/colour.py"
    )
