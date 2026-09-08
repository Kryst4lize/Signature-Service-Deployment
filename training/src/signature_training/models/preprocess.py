"""The input contract for the VGG16 / ResNet50 extractors.

Everything that feeds an extractor goes through `extractor_preprocess`, in
training and in evaluation alike. There were three call sites doing this
independently — `train/verification.py` twice and `evaluate/runner.py` once —
and all three did the same thing, which is exactly the arrangement in which one
of them later stops doing the same thing.

Two steps, in this order:

    1. colour.apply(mode)   paper white-balance, in [0, 255] HWC
    2. preprocess_input     Keras "caffe" mode: RGB->BGR, x255 is already done,
                            subtract [103.939, 116.779, 123.68]

Order matters. The white-balance estimates the paper level from the image, and
after Caffe preprocessing there is no paper level to estimate — the channels are
reordered, mean-subtracted and signed. Correcting afterwards would be estimating
a "paper colour" from a tensor whose brightest region is no longer paper.

WHY THE COLOUR STEP IS HERE AT ALL

The backbones are trained on raw Kaggle scans, whose paper measures
R 243.28 / G 251.47 / B 251.46 (B-R = +8.18, a red deficit that reads as cyan).
At serving time they are handed CycleGAN output, whose paper measures
249.98 / 253.43 / 253.43 (B-R = +3.45). Two different colour distributions, and
until now neither side normalised, so the embedding had to absorb a difference
that has nothing to do with whose signature it is.

Normalising the paper to a common white point at the moment of entry removes
that difference on both sides. It costs three multiplies.

WHERE IT IS APPLIED, AND WHY EXACTLY ONCE

`data/cyclegan.py:build_verification_split` corrects the images as it writes
them, so what is on disk under `paths.verification_dataset` is already what the
backbones should see. Everything reading that dataset therefore passes
mode="none" here and gets Caffe preprocessing alone.

Correcting at build time rather than in this hook is not a preference. Keras only
offers a POST-augmentation hook: `preprocessing_function` runs after
rotation/shift/shear/zoom, so it sees the borders those fill in. The configured
augmentation fills a mean of 10.9% of the frame (p95 17.3%, max 20.8%, over 300
draws) with `cval=255`; those neutral pixels land inside the paper band and drag
the per-channel means together, weakening the correction from +8.17 -> +0.00
down to +8.17 -> +0.83. At build time there is no augmentation and the estimate
is clean.

And applying it in BOTH places would be unsafe, which is worth stating because
it looks free. whiten() is idempotent only while its gain clamp does not bind.
The clamp is what breaks idempotence, not what guarantees it: when
`estimate_paper(img).max() / .min()` exceeds `max_gain`, the first pass is
truncated and by construction leaves the paper un-neutral, so a second pass
applies the remainder. Two passes then reach `max_gain**2` = 2.56, silently
doubling the safety limit colour.py sets to stop a dark photograph being
stretched into white. Measured on a tungsten capture with paper (248, 190, 130):
one pass caps blue at 1.600x as intended, two reach 1.908x — 41 levels of
per-pixel drift and a VGG16 embedding cosine distance of 0.028.

The real scans never approach that (imbalance 1.034 raw, 1.014 denoised), so the
composition happens to be safe on this corpus. It is not safe in general, and the
serving side must therefore apply the correction exactly once too.

THE SERVING HALF IS NOT WIRED YET

The service declares COLOUR_MODE and no code reads it. When it is wired, the
correction belongs immediately before `to_caffe` in
`inference/api/app/triton.py` — once, on the denoised tensor. `data/colour.py`
is byte-identical to the service's copy for exactly this reason.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..data import colour

# Imported lazily inside the factory. This module is imported by `evaluate` and
# by config-only code paths, and pulling TensorFlow in at import time costs
# several seconds and a great deal of memory for callers that never build a model.
_KERAS_PREPROCESS: dict[str, Callable] = {}


def _keras_preprocess(backbone: str) -> Callable:
    if not _KERAS_PREPROCESS:
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet
        from tensorflow.keras.applications.vgg16 import preprocess_input as vgg

        _KERAS_PREPROCESS.update({"vgg16": vgg, "resnet50": resnet})
    try:
        return _KERAS_PREPROCESS[backbone]
    except KeyError:
        raise ValueError(
            f"Unknown backbone {backbone!r}; expected one of {sorted(_KERAS_PREPROCESS)}"
        ) from None


def extractor_preprocess(backbone: str, mode: str = "none") -> Callable[[np.ndarray], np.ndarray]:
    """Return the function that turns a loaded image into extractor input.

    The returned callable takes ONE image as HWC float in [0, 255] and returns
    the same shape, Caffe-preprocessed. That is precisely the contract Keras'
    `ImageDataGenerator(preprocessing_function=...)` expects — it calls the
    function per image, after resizing and augmentation and before batching — so
    the same object can be handed to a generator or called directly on a single
    array.

    `mode` is for callers holding a RAW image. Anything read from
    `paths.verification_dataset` was already corrected when the dataset was
    written, so the pipeline passes "none" — see the module docstring for why
    correcting twice is not free. "none" makes step 1 an identity, byte-for-byte
    what this did before the contract existed.
    """
    if mode not in {"none", "whiten", "desaturate"}:
        raise ValueError(f"Unknown colour mode {mode!r}; expected none|whiten|desaturate")
    caffe = _keras_preprocess(backbone)

    def preprocess(img: np.ndarray) -> np.ndarray:
        corrected = colour.apply(np.asarray(img, dtype=np.float32), mode)
        # A copy, because preprocess_input(mode="caffe") MUTATES ITS ARGUMENT.
        # It reverses the channel axis with `x[..., ::-1]`, which numpy returns
        # as a view, then subtracts the mean into that view — so the caller's
        # array comes back modified. Verified on TF 2.21: passing a 200.0-filled
        # array leaves it holding [76.32, 83.22, 96.06]. Without the copy, an
        # array the caller still intends to use is silently corrupted.
        return caffe(np.array(corrected, dtype=np.float32, copy=True))

    preprocess.__doc__ = f"{backbone} extractor input: colour={mode!r} then Keras caffe."
    return preprocess


def embed_image(
    extractor,
    path,
    backbone: str,
    mode: str,
    size: int = 224,
) -> np.ndarray:
    """One image file -> L2-normalised embedding.

    Shared by `train/verification.py:embed` and `evaluate/runner.py`, which
    previously carried two copies of this that were identical and had no
    mechanism to stay that way.
    """
    from tensorflow.keras.preprocessing import image as keras_image

    img = keras_image.load_img(str(path), target_size=(size, size))
    arr = extractor_preprocess(backbone, mode)(keras_image.img_to_array(img))
    vec = extractor.predict(np.expand_dims(arr, 0), verbose=0).flatten()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec
