"""Tests for the tensor conventions between the API and each model.

These are the contracts that were silently violated before the refactor: every
model still returned a well-formed tensor, it was just computed on
out-of-distribution input, so nothing failed loudly.
"""

import numpy as np
import pytest

from app.images import pil_to_tensor, tensor_to_pil
from app.triton import (
    _CAFFE_MEAN_BGR,
    from_cyclegan,
    l2_normalise,
    to_caffe,
    to_cyclegan,
)


# ── Base convention: [1, 3, H, W] float32 in [0, 1], RGB ──────────────────────


def test_pil_to_tensor_shape_range_and_channel_order(rgb_image):
    t = pil_to_tensor(rgb_image, (224, 224))

    assert t.shape == (1, 3, 224, 224)
    assert t.dtype == np.float32
    assert 0.0 <= t.min() and t.max() <= 1.0

    # Left third is red: channel 0 high, channels 1 and 2 low.
    left = t[0, :, :, :50]
    assert left[0].mean() > 0.9
    assert left[1].mean() < 0.1
    assert left[2].mean() < 0.1

    # Right third is blue: channel 2 high.
    right = t[0, :, :, -50:]
    assert right[2].mean() > 0.9
    assert right[0].mean() < 0.1


def test_tensor_to_pil_round_trips(rgb_image):
    t = pil_to_tensor(rgb_image, (64, 64))
    back = pil_to_tensor(tensor_to_pil(t), (64, 64))
    # Only 8-bit quantisation should differ.
    assert np.abs(t - back).max() < 1.0 / 255 + 1e-6


# ── CycleGAN: [-1, 1] in and out ──────────────────────────────────────────────


def test_to_cyclegan_maps_unit_range_to_symmetric_range():
    x = np.array([[[[0.0, 0.5, 1.0]]]], dtype=np.float32)
    np.testing.assert_allclose(to_cyclegan(x), [[[[-1.0, 0.0, 1.0]]]], atol=1e-6)


def test_from_cyclegan_inverts_to_cyclegan(unit_tensor):
    np.testing.assert_allclose(
        from_cyclegan(to_cyclegan(unit_tensor)), unit_tensor, atol=1e-6
    )


def test_from_cyclegan_recovers_the_negative_half():
    """The bug this guards: treating the Tanh output as if it were already
    [0, 1] sent every negative value through `.clip(0, 255)`, collapsing the
    darker half of the denoised signature to solid black."""
    tanh_out = np.array([[[[-1.0, -0.5, 0.0, 0.5, 1.0]]]], dtype=np.float32)
    got = from_cyclegan(tanh_out)

    np.testing.assert_allclose(got, [[[[0.0, 0.25, 0.5, 0.75, 1.0]]]], atol=1e-6)
    # Every input value maps somewhere distinct - nothing is crushed to 0.
    assert len(np.unique(got)) == 5


def test_from_cyclegan_clips_out_of_range_output():
    noisy = np.array([[[[-1.4, 1.4]]]], dtype=np.float32)
    got = from_cyclegan(noisy)
    assert got.min() >= 0.0 and got.max() <= 1.0


# ── Extractors: Keras preprocess_input(mode="caffe") ──────────────────────────


def test_to_caffe_swaps_to_bgr_and_subtracts_imagenet_mean():
    """Reference values come from the definition of
    keras.applications.imagenet_utils.preprocess_input(mode="caffe"):
    RGB -> BGR, scale to [0, 255], subtract [103.939, 116.779, 123.68].
    """
    # One pure-red pixel in [0, 1] RGB.
    red = np.zeros((1, 3, 1, 1), dtype=np.float32)
    red[0, 0, 0, 0] = 1.0

    got = to_caffe(red)[0, :, 0, 0]

    # After the swap red lands in the last (R) channel of a BGR tensor.
    np.testing.assert_allclose(got, [-103.939, -116.779, 255.0 - 123.68], atol=1e-4)


def test_to_caffe_black_is_the_negated_mean():
    black = np.zeros((1, 3, 2, 2), dtype=np.float32)
    got = to_caffe(black)
    expected = -_CAFFE_MEAN_BGR
    np.testing.assert_allclose(got, np.broadcast_to(expected, got.shape), atol=1e-4)


def test_to_caffe_output_is_not_unit_range(unit_tensor):
    """Guards the original defect directly: the extractors were being fed
    values in [0, 1] when they expect roughly [-124, +151]."""
    got = to_caffe(unit_tensor)
    assert got.min() < -50.0
    assert got.max() > 50.0
    assert got.dtype == np.float32


def test_to_caffe_is_contiguous_after_the_channel_reversal(unit_tensor):
    """`x[:, ::-1]` produces a negative-stride view; tritonclient needs a real
    C-contiguous buffer."""
    got = np.ascontiguousarray(to_caffe(unit_tensor))
    assert got.flags["C_CONTIGUOUS"]


# ── Embeddings ────────────────────────────────────────────────────────────────


def test_l2_normalise_produces_unit_vectors():
    vec = np.array([3.0, 4.0], dtype=np.float32)
    got = l2_normalise(vec)
    np.testing.assert_allclose(got, [0.6, 0.8], atol=1e-6)
    assert np.isclose(np.linalg.norm(got), 1.0)


def test_l2_normalise_handles_the_zero_vector():
    """A fully dead ReLU tap must not produce NaNs that poison the column."""
    got = l2_normalise(np.zeros(16, dtype=np.float32))
    assert not np.isnan(got).any()
    assert np.all(got == 0.0)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_l2_normalise_makes_cosine_distance_scale_invariant(scale):
    """Why normalisation matters: raw fc1 magnitudes vary with input contrast,
    so an un-normalised distance threshold is not comparable across images."""
    rng = np.random.default_rng(1)
    vec = rng.random(64).astype(np.float32)
    a = l2_normalise(vec)
    b = l2_normalise(vec * scale)
    np.testing.assert_allclose(a, b, atol=1e-5)
