"""The colour normalisation contract.

`colour.mode` is one setting governing every point an image enters a model. The
tests here pin the properties that make that safe, and each one corresponds to a
way the contract could silently stop holding:

  * "none" is an exact identity           -> turning the feature off is free
  * whiten is idempotent                  -> applying it twice cannot hurt
  * the Keras hook signature is what we think it is
  * augmentation fill degrades the estimate -> why build-time correction exists
  * an old config names the new key
  * a dataset built under one mode is not silently extended under another
"""

from __future__ import annotations

import numpy as np
import pytest

from signature_training.config import Config
from signature_training.data import colour, cyclegan

# Measured on the real corpora. See documentation/02-pipeline-deep-dive.md.
RAW_SCAN_PAPER = (243.28, 251.47, 251.46)  # Kaggle, 400 genuine images: B-R +8.18
DENOISED_PAPER = (249.98, 253.43, 253.43)  # real latest_net_G_B output: B-R +3.45
INK = (102.15, 89.88, 95.67)

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def page(paper, ink=INK, size=224, ink_frac=0.15, seed=0):
    """A synthetic scan: mostly paper at the given cast, a minority of ink."""
    img = np.tile(np.asarray(paper, np.float32), (size, size, 1))
    rng = np.random.default_rng(seed)
    img += rng.normal(scale=1.0, size=img.shape).astype(np.float32)
    n = int(size * size * ink_frac)
    img[rng.integers(0, size, n), rng.integers(0, size, n)] = np.asarray(ink, np.float32)
    return np.clip(img, 0, 255)


def cast(img):
    """B - R over the paper region: the number the whole exercise is about."""
    bg = img[img @ LUMA > 200]
    m = bg.mean(axis=0)
    return float(m[2] - m[0])


# ── the properties that make the contract safe ────────────────────────────────


def test_mode_none_is_an_exact_identity():
    """The default must not perturb a single value, or enabling the contract
    would silently change results for everyone who left it off."""
    img = page(RAW_SCAN_PAPER)
    np.testing.assert_array_equal(colour.apply(img, "none"), img)


@pytest.mark.parametrize(
    ("name", "paper"), [("raw scan", RAW_SCAN_PAPER), ("denoised", DENOISED_PAPER)]
)
def test_whiten_is_idempotent(name, paper):
    """Why it is safe to correct at dataset-build time AND again at the Keras
    hook: whiten clamps its gain to [1.0, max_gain], so neutral paper yields
    exactly 1.0. Measured second-pass residual: 0.001 levels on the raw cast,
    0.026 on the denoised one."""
    once = colour.whiten(page(paper))
    twice = colour.whiten(once)
    assert np.abs(twice - once).max() < 0.05, f"{name}: not idempotent"
    assert abs(cast(twice)) <= abs(cast(once)) + 1e-3


@pytest.mark.parametrize(
    ("name", "paper"), [("raw scan", RAW_SCAN_PAPER), ("denoised", DENOISED_PAPER)]
)
def test_whiten_neutralises_the_measured_casts(name, paper):
    img = page(paper)
    assert cast(img) > 3.0, f"{name}: fixture does not carry the cast"
    assert abs(cast(colour.whiten(img))) < 0.2, name


def test_augmentation_fill_degrades_the_estimate():
    """The measurement that put the correction into the dataset builder.

    Keras runs `preprocessing_function` AFTER augmentation, so it sees borders
    filled with cval=255. Those neutral pixels join the paper band and pull the
    channel means together. At the mean fill fraction the configured
    augmentation produces (10.9%), roughly a tenth of the cast survives.
    """
    img = page(RAW_SCAN_PAPER)
    clean_residual = abs(cast(colour.whiten(img)))

    filled = img.copy()
    rng = np.random.default_rng(1)
    n = int(224 * 224 * 0.109)
    filled[rng.integers(0, 224, n), rng.integers(0, 224, n)] = 255.0
    filled_residual = abs(cast(colour.whiten(filled)))

    assert clean_residual < 0.2
    assert filled_residual > 4 * clean_residual, (
        "fill no longer degrades the estimate - if the estimator changed, the "
        "build-time correction in build_verification_split may be redundant"
    )


# ── config ────────────────────────────────────────────────────────────────────


def test_colour_mode_defaults_to_none():
    assert Config().colour.mode == "none"


def test_default_yaml_parses_and_exposes_colour():
    from signature_training.paths import PROJECT_ROOT

    cfg = Config.load(PROJECT_ROOT / "configs" / "default.yaml")
    assert cfg.colour.mode in {"none", "whiten", "desaturate"}


def test_old_config_key_names_the_new_one(tmp_path):
    """A config written before the rename must fail with the replacement, not
    with a bare 'unknown key' - and above all must not silently stop
    correcting."""
    stale = tmp_path / "stale.yaml"
    stale.write_text("cyclegan_data:\n  image_size: 512\n  colour_mode: whiten\n")
    with pytest.raises(ValueError, match=r"colour\.mode"):
        Config.load(stale)


def test_override_reaches_the_new_section():
    cfg = Config.load(overrides={"colour.mode": "whiten"})
    assert cfg.colour.mode == "whiten"


# ── the dataset stamp ─────────────────────────────────────────────────────────


def test_stamp_is_written_and_accepted_on_rerun(tmp_path):
    cyclegan._check_colour_stamp(tmp_path, "whiten")
    assert (tmp_path / cyclegan.COLOUR_STAMP).read_text().strip() == "whiten"
    cyclegan._check_colour_stamp(tmp_path, "whiten")  # idempotent


def test_changing_mode_on_an_existing_dataset_is_refused(tmp_path):
    """build_verification_split skips folders that already exist, so a changed
    mode would otherwise leave half the corpus under the old distribution."""
    cyclegan._check_colour_stamp(tmp_path, "none")
    with pytest.raises(RuntimeError, match="mix two colour distributions"):
        cyclegan._check_colour_stamp(tmp_path, "whiten")


# ── end to end through the dataset builder ────────────────────────────────────


def _cast_corpus(root, paper=RAW_SCAN_PAPER):
    """A miniature Kaggle tree whose paper carries the real measured cast."""
    from PIL import Image

    for split, people in (("train", ["001", "002"]), ("test", ["003"])):
        for person in people:
            for suffix in ("", "_forg"):
                d = root / split / f"{person}{suffix}"
                d.mkdir(parents=True)
                for i in range(2):
                    a = page(paper, size=64, ink_frac=0.10, seed=i).astype(np.uint8)
                    Image.fromarray(a).save(d / f"{person}_{i}.png")
    return root


def _built(tmp_path, mode):
    dst = tmp_path / "verification"
    cfg = Config.load(
        overrides={
            "paths.raw_signatures": str(_cast_corpus(tmp_path / "raw")),
            "paths.verification_dataset": str(dst),
            "colour.mode": mode,
        }
    )
    cyclegan.build_verification_split(cfg)
    return dst


def _on_disk_cast(dataset):
    from PIL import Image

    files = sorted(dataset.rglob("*.png"))
    assert files, f"nothing written under {dataset}"
    return np.mean([cast(np.asarray(Image.open(f), np.float32)) for f in files])


def test_verification_split_writes_uncorrected_images_by_default(tmp_path):
    assert _on_disk_cast(_built(tmp_path, "none")) > 7.0


def test_verification_split_writes_corrected_images_when_asked(tmp_path):
    """The end of the wire: `colour.mode` set in the config must change the
    pixels the backbones actually read off disk."""
    assert abs(_on_disk_cast(_built(tmp_path, "whiten"))) < 1.0


def test_verification_split_records_the_mode_it_used(tmp_path):
    dst = _built(tmp_path, "whiten")
    assert (dst / cyclegan.COLOUR_STAMP).read_text().strip() == "whiten"


# ── the Keras side ────────────────────────────────────────────────────────────
# Split out because these import TensorFlow, which is slow and is not needed for
# anything above.

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")


def _keras_caffe(backbone):
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg

    return vgg if backbone == "vgg16" else resnet


@pytest.mark.parametrize("backbone", ["vgg16", "resnet50"])
def test_mode_none_is_byte_identical_to_plain_keras(backbone):
    """The default path must be exactly what it was before this module existed.
    Anything else is a silent change to every model already trained."""
    from signature_training.models.preprocess import extractor_preprocess

    img = page(RAW_SCAN_PAPER, size=64)
    got = extractor_preprocess(backbone, "none")(img)
    want = _keras_caffe(backbone)(img.copy())
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("backbone", ["vgg16", "resnet50"])
def test_preprocess_does_not_mutate_its_input(backbone):
    """preprocess_input(mode="caffe") reverses the channel axis into a numpy
    VIEW and subtracts the mean in place, so it modifies the caller's array.
    ImageDataGenerator would not notice; a caller reusing the array would."""
    from signature_training.models.preprocess import extractor_preprocess

    img = page(RAW_SCAN_PAPER, size=32)
    before = img.copy()
    extractor_preprocess(backbone, "whiten")(img)
    np.testing.assert_array_equal(img, before)


def test_whiten_then_caffe_differs_from_caffe_alone():
    """Guards against the correction being wired in but doing nothing - the
    failure mode that made two earlier implementations no-ops."""
    from signature_training.models.preprocess import extractor_preprocess

    img = page(RAW_SCAN_PAPER, size=64)
    plain = extractor_preprocess("vgg16", "none")(img)
    corrected = extractor_preprocess("vgg16", "whiten")(img)
    assert np.abs(corrected - plain).max() > 1.0


def test_rejects_an_unknown_mode():
    from signature_training.models.preprocess import extractor_preprocess

    with pytest.raises(ValueError, match=r"none\|whiten\|desaturate"):
        extractor_preprocess("vgg16", "sepia")


def test_keras_hook_receives_one_hwc_image_in_0_255(tmp_path):
    """The contract extractor_preprocess is written against. If a future Keras
    batches the call or pre-scales to [0, 1], the correction silently operates
    on the wrong domain and every paper estimate is wrong."""
    from PIL import Image
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    for person in ("001", "002"):
        (tmp_path / person).mkdir()
        for i in range(3):
            a = np.full((40, 60, 3), 243, np.uint8)
            a[..., 1:] = 251
            a[10:20, 10:30] = 90
            Image.fromarray(a).save(tmp_path / person / f"{i}.png")

    seen = []

    def spy(img):
        seen.append(img)
        return img

    flow = ImageDataGenerator(
        preprocessing_function=spy, validation_split=0.34
    ).flow_from_directory(
        directory=str(tmp_path),
        target_size=(24, 24),
        color_mode="rgb",
        batch_size=2,
        class_mode="categorical",
        seed=7,
        subset="training",
    )
    next(flow)

    assert seen, "preprocessing_function was never called"
    for img in seen:
        assert img.ndim == 3 and img.shape[-1] == 3, f"not one HWC image: {img.shape}"
    assert max(float(i.max()) for i in seen) > 2.0, "input is not in [0, 255]"
