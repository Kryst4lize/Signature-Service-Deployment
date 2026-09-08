"""Paper white-balance.

The Kaggle signature scans carry a systematic colour cast. Measured over 400
genuine training images (background = pixels with luminance > 200):

    R = 243.28   G = 251.47   B = 251.46      B - R = +8.18

Green and blue are equal to within 0.01, so nothing is adding blue — red is
*missing*, by about 12 levels off pure white, and a red deficit reads as cyan.
The cast is essentially constant across the corpus (per-image B-R has sd 0.15,
range +7.83..+8.94), which says one scanner and one fixed offset rather than
per-image variation.

That is why a diagonal (von Kries) correction is the right tool: estimate the
paper level per channel and scale each channel so paper becomes 255. It is a
three-multiply operation, it is exactly invertible, and because the cast is
uniform it removes essentially all of it.

CONTRACT: this is a byte-for-byte mirror of
training/src/signature_training/data/colour.py. The two halves are independent
packages with no shared import — the same arrangement as the Caffe
preprocessing — and tests/test_colour.py pins them against each other so they
cannot drift.
"""

from __future__ import annotations

import numpy as np

# Only ever brighten, and never by more than this. A dark photograph is not a
# tinted scan, and stretching it to white would destroy the strokes.
DEFAULT_MAX_GAIN = 1.6

# Fraction of the image treated as "not paper" when estimating the paper colour.
# The brightest (1 - this) of pixels are taken to be paper, and their per-channel
# MEDIAN is the estimate.
#
# This started as a plain 99th percentile, which was wrong in the case that
# matters most. On the raw scans it works — paper sits at 243/251, so p99 lands
# on it. On the DENOISER'S OUTPUT it does not: the generator pushes paper up
# against 255, so p99 saturates at 255 in all three channels, the computed gain
# comes out ~1.0, and the correction silently does nothing. Measured on the real
# latest_net_G_B: paper B-R +3.46 before, +3.45 after. A median over the paper
# bulk sees the actual level in both cases.
# Paper is everything at or above this fraction of `white` in luminance. A
# relative threshold, so it survives a globally darker scan.
DEFAULT_PAPER_LEVEL = 0.78


def estimate_paper(
    img: np.ndarray, paper_level: float = DEFAULT_PAPER_LEVEL, white: float = 255.0
) -> np.ndarray:
    """Per-channel paper level of an HxWx3 image, in the image's own units.

    The MEAN over the paper region, not a high percentile and not a median of
    the brightest pixels. Both of those were tried and both failed on the
    denoiser's own output, where paper occupies a band from ~249 up to a
    saturated spike at 255: a percentile or a top-40% median lands on the spike,
    reports ~255 for every channel, and the correction becomes a no-op. The mean
    over the whole paper band sees the real 249.97 / 253.44 / 253.43.
    """
    flat = img.reshape(-1, img.shape[-1])
    lum = flat @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    paper = flat[lum >= paper_level * white]
    if len(paper) < 32:  # almost no background to go on
        thresh = np.quantile(lum, 0.75)
        paper = flat[lum >= thresh]
    if len(paper) < 1:
        return np.full(img.shape[-1], white, dtype=np.float32)
    return paper.mean(axis=0)


def whiten(
    img: np.ndarray,
    paper_level: float = DEFAULT_PAPER_LEVEL,
    strength: float = 1.0,
    max_gain: float = DEFAULT_MAX_GAIN,
    white: float = 255.0,
    lift: bool = False,
) -> np.ndarray:
    """Neutralise a paper colour cast by per-channel gain.

    Parameters
    ----------
    img        HxWx3, float or uint8, in [0, white].
    strength   0 = no change, 1 = paper mapped exactly to `white`. Values in
               between are useful if a partial correction is preferred to a
               full one.
    max_gain   Upper clamp per channel.
    white      Saturation value (255 for uint8, 1.0 for unit floats).
    lift       Also brighten paper toward `white`, not just neutralise it.

    Returns the corrected image in the input dtype's range, as float32.
    """
    src = img.astype(np.float32)
    paper = estimate_paper(src, paper_level, white).astype(np.float32)

    # Equalise the channels against the BRIGHTEST one rather than driving them
    # all to `white`. Targeting white looks equivalent and is not: paper often
    # already sits within a couple of levels of saturation, so the extra gain is
    # swallowed by the clip at `white` and the cast survives. Scaling to
    # paper.max() removes the cast without asking any channel to exceed a value
    # it already reaches. `lift` re-adds the brightness push when wanted.
    target = paper.max() if not lift else white
    gain = target / np.maximum(paper, 1e-6)
    gain = np.clip(gain, 1.0, max_gain)  # brighten only
    gain = 1.0 + strength * (gain - 1.0)

    return np.clip(src * gain, 0.0, white).astype(np.float32)


def desaturate(img: np.ndarray, white: float = 255.0) -> np.ndarray:
    """Collapse to luminance, replicated across three channels.

    The stronger option: a signature's identity is in the stroke geometry, not
    its colour, so discarding chroma removes the cast AND every other
    scanner-dependent colour difference at once. It is a bigger change than
    `whiten` — the models are ImageNet-pretrained and do use colour — so it is
    offered rather than assumed.
    """
    src = img.astype(np.float32)
    lum = src @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.clip(np.repeat(lum[..., None], 3, axis=-1), 0.0, white).astype(np.float32)


def apply(img: np.ndarray, mode: str, **kwargs) -> np.ndarray:
    """Dispatch on a config string: "none" | "whiten" | "desaturate"."""
    if mode == "none":
        return img.astype(np.float32)
    if mode == "whiten":
        return whiten(img, **kwargs)
    if mode == "desaturate":
        return desaturate(img, white=kwargs.get("white", 255.0))
    raise ValueError(f"Unknown colour mode {mode!r}; expected none|whiten|desaturate")
