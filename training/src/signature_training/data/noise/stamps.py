"""
stamp_augmentation.py
─────────────────────────────────────────────────────────────────────────────
Realistic stamp / seal noise augmentation for CycleGAN signature cleaning.

Key realism improvements over a naïve overlay
──────────────────────────────────────────────
1. DPI-aware scaling
   Real stamps are 38-42 mm diameter.  Scanned documents come in at
   150-300 DPI.  After Phase-1 crops the signature bbox the stamp appears
   large relative to the crop.  We model this explicitly:

       stamp_px ≈ stamp_mm_diameter × (scan_dpi / 25.4)
       scale    = stamp_px / max(crop_h, crop_w)

   Folder stamps can be 1000+ px — they are scaled DOWN to match.

2. Partial visibility
   Phase-1 crops tightly.  The stamp, positioned left-of-signature,
   usually bleeds beyond the left/top/bottom edge.  Only a right-side arc
   of the round seal is visible (see real documents in earlier images).

3. Red ink simulation
   Greyscale / black stamp scans are tinted to the Vietnamese standard
   red ink colour (H≈0°, S≈85%, V≈90%).

4. Ink bleed / transparency jitter
   Faded stamps (older ink) are reproduced by random opacity.
"""

import glob
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np



# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Vietnamese standard round stamp diameter range (mm)
STAMP_DIAM_MM_MIN: float = 36.0
STAMP_DIAM_MM_MAX: float = 42.0

# Document scan resolution range (DPI).
# 150 = low-quality scan | 300 = typical office scanner
SCAN_DPI_MIN: int = 150
SCAN_DPI_MAX: int = 300

# Proportion of stamp diameter relative to the CROP SHORT-SIDE that we allow
# to float when we don't know the actual scan DPI (fallback mode).
STAMP_CROP_RATIO_MIN: float = 0.80
STAMP_CROP_RATIO_MAX: float = 1.80


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_achromatic(bgr: np.ndarray, mask: np.ndarray) -> bool:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(cv2.mean(hsv[:, :, 1], mask=mask)[0])
    return mean_sat < 35


def _tint_red(bgra: np.ndarray, strength: float = 0.60) -> np.ndarray:
    """Push ink colour toward Vietnamese official red (#D0021B-ish)."""
    out = bgra.astype(np.float32)
    # Target: approximately (B=30, G=20, R=210)
    out[:, :, 2] = np.clip(out[:, :, 2] + strength * 160, 0, 255)   # R ↑
    out[:, :, 1] = np.clip(out[:, :, 1] * (1 - strength * 0.8), 0, 255)  # G ↓
    out[:, :, 0] = np.clip(out[:, :, 0] * (1 - strength * 0.8), 0, 255)  # B ↓
    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Stamp preparation
# ─────────────────────────────────────────────────────────────────────────────

def _remove_background(bgr: np.ndarray) -> np.ndarray:
    """Return BGRA with white/near-white pixels made transparent."""
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    bgra[:, :, 3] = cv2.bitwise_not(white_mask)
    return bgra


def _load_stamps(folder: str) -> List[np.ndarray]:
    """
    Load every stamp image in `folder` as a BGRA array, ready for compositing.
    White background is removed; achromatic (greyscale/black) stamps are
    tinted red to match Vietnamese ink regulations.
    """
    exts = [e for base in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for e in (base, base.upper())]
    paths: List[str] = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = list(set(paths))

    stamps: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        bgra = _remove_background(img)
        alpha_mask = bgra[:, :, 3]
        if _is_achromatic(img, alpha_mask):
            bgra = _tint_red(bgra)
        stamps.append(bgra)

    return stamps


def _rotate_bgra(bgra: np.ndarray, angle: float) -> np.ndarray:
    """Rotate BGRA keeping full content (expands canvas)."""
    h, w = bgra.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return cv2.warpAffine(
        bgra, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Compositing
# ─────────────────────────────────────────────────────────────────────────────

def _composite(
    bg: np.ndarray,
    stamp_bgra: np.ndarray,
    x: int,
    y: int,
    opacity: float,
) -> np.ndarray:
    bg_h, bg_w = bg.shape[:2]
    st_h, st_w = stamp_bgra.shape[:2]

    sx0 = max(0, -x);      sy0 = max(0, -y)
    sx1 = min(st_w, bg_w - x)
    sy1 = min(st_h, bg_h - y)

    if sx1 <= sx0 or sy1 <= sy0:
        return bg

    dx0 = x + sx0;  dy0 = y + sy0
    dx1 = x + sx1;  dy1 = y + sy1

    crop  = stamp_bgra[sy0:sy1, sx0:sx1]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0 * opacity

    fg     = crop[:, :, :3].astype(np.float32) / 255.0   # stamp  [0,1]
    bg_roi = bg[dy0:dy1, dx0:dx1].astype(np.float32) / 255.0  # bg [0,1]

    # ── Multiply blend: signature ink wins over stamp ink ──────────────────
    multiplied = fg * bg_roi                               # both darken each other

    # Blend between original background and the multiply result
    # using stamp alpha as the mix weight — stamp only shows where it has ink
    blended = (alpha * multiplied + (1.0 - alpha) * bg_roi) * 255.0

    out = bg.copy()
    out[dy0:dy1, dx0:dx1] = blended.astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scale logic  (DPI-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_target_stamp_px(crop_h: int, crop_w: int, rng: random.Random) -> int:
    """
    Estimate a realistic stamp diameter in pixels for this crop size.

    Method
    ------
    We simulate a random scan DPI, compute how many pixels a standard
    Vietnamese round stamp would occupy, then return that value —
    regardless of how large the source stamp image actually is.
    This prevents high-res folder stamps from dominating the crop.
    """
    dpi     = rng.randint(SCAN_DPI_MIN, SCAN_DPI_MAX)
    diam_mm = rng.uniform(STAMP_DIAM_MM_MIN, STAMP_DIAM_MM_MAX)
    diam_px = diam_mm * dpi / 25.4          # convert mm → pixels at that DPI

    # The crop itself was extracted from the same scan, so we need the
    # stamp diameter expressed in *crop* pixels.
    # We treat the crop short-side as our reference: a signature crop of
    # height H came from a region whose physical size ≈ H * 25.4 / dpi mm.
    # The stamp should be diam_mm of that region.
    short = min(crop_h, crop_w)
    crop_mm = short * 25.4 / dpi
    # fraction of crop covered by stamp diameter
    frac = diam_mm / crop_mm
    target_px = int(short * frac)

    # Safety clamp: stamp cannot be smaller than 40 px or 3× the crop height
    return max(40, min(target_px, 3 * crop_h))


# ─────────────────────────────────────────────────────────────────────────────
# Placement logic  (Vietnamese rule)
# ─────────────────────────────────────────────────────────────────────────────

def _placement(
    crop_h: int,
    crop_w: int,
    st_h: int,
    st_w: int,
    p_partial: float,
    rng: random.Random,
) -> Tuple[int, int]:
    """
    Return (x, y) top-left for stamp placement.

    Vietnamese rule: stamp covers left ~1/3 of signature.
    Phase-1 crops tightly → stamp usually bleeds off left edge.

    Three modes
    ───────────
    A  Partial-left  (most common): right arc visible, left off-frame
    B  Corner bleed             : left + top or bottom off-frame
    C  Full stamp               : entirely inside crop (less common)
    """
    draw = rng.random()

    if draw < p_partial * 0.55:
        # ── A: partial left ────────────────────────────────────────────────
        vis = rng.uniform(0.15, 0.55)
        x = -int(st_w * (1.0 - vis))
        cy = int(crop_h * rng.uniform(0.30, 0.70))
        y = cy - st_h // 2

    elif draw < p_partial:
        # ── B: left + vertical bleed ───────────────────────────────────────
        vis = rng.uniform(0.15, 0.50)
        x = -int(st_w * (1.0 - vis))
        if rng.random() < 0.5:
            y = -int(st_h * rng.uniform(0.25, 0.55))   # top bleed
        else:
            y = int(crop_h - st_h * rng.uniform(0.45, 0.75))  # bottom

    else:
        # ── C: full stamp, left-biased ─────────────────────────────────────
        x = rng.randint(0, max(1, int(crop_w * 0.30)))
        cy = int(crop_h * rng.uniform(0.20, 0.65))
        y = cy - st_h // 2

    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class StampAugmentor:
    """
    Add realistic Vietnamese stamp / seal noise to a signature crop.

    Parameters
    ----------
    stamp_folder  : Directory containing stamp images.
    p_apply       : Probability a stamp is added at all.
    p_partial     : Among applied stamps, probability the stamp bleeds out
                    of frame (simulating tight Phase-1 crop).
    opacity_range : (min, max) compositing opacity.
    angle_range   : (min, max) rotation degrees.
    """

    def __init__(
        self,
        stamp_folder: str = "stamp_noise",
        p_apply: float = 0.70,
        p_partial: float = 0.68,
        opacity_range: Tuple[float, float] = (0.50, 0.92),
        angle_range: Tuple[float, float] = (-12.0, 12.0),
        rng: Optional[random.Random] = None,
    ) -> None:
        self.p_apply        = p_apply
        self.p_partial      = p_partial
        self.opacity_range  = opacity_range
        self.angle_range    = angle_range
        # Owned by the caller so one seed governs the whole dataset build.
        # Using the module-level `random` made stamp placement depend on
        # whatever else in the process had consumed from it.
        self.rng            = rng if rng is not None else random.Random()
        self._stamps        = _load_stamps(stamp_folder)

        if not self._stamps:
            import warnings
            warnings.warn(
                f"StampAugmentor: no stamp images found in '{stamp_folder}'.",
                stacklevel=2,
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self._stamps or self.rng.random() > self.p_apply:
            return image

        h, w = image.shape[:2]
        stamp = self._prepare(h, w)
        x, y  = _placement(h, w, stamp.shape[0], stamp.shape[1], self.p_partial, self.rng)
        opacity = self.rng.uniform(*self.opacity_range)
        return _composite(image, stamp, x, y, opacity)

    # ------------------------------------------------------------------
    def _prepare(self, crop_h: int, crop_w: int) -> np.ndarray:
        """
        Scale and rotate a random stamp so its size is physically realistic
        relative to this particular signature crop.
        """
        raw = self.rng.choice(self._stamps).copy()
        raw_h, raw_w = raw.shape[:2]

        # DPI-aware target diameter in pixels
        target_diam_px = _compute_target_stamp_px(crop_h, crop_w, self.rng)

        # Scale the stamp so its SHORT side equals target_diam_px
        # (stamps are roughly circular so either dimension works)
        scale = target_diam_px / min(raw_h, raw_w)
        new_h = max(20, int(raw_h * scale))
        new_w = max(20, int(raw_w * scale))
        stamp = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Small rotation jitter
        angle = self.rng.uniform(*self.angle_range)
        if abs(angle) > 0.5:
            stamp = _rotate_bgra(stamp, angle)

        return stamp
