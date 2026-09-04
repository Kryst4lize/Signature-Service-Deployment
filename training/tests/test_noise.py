"""Augmentation tests.

The first two are regression tests for the defect that made the whole CycleGAN
dataset degenerate: an RNG re-seeded with a fixed literal inside the per-image
noise function, so every generated image received identical noise.
"""

import numpy as np
import pytest

from signature_training.data.noise.document import DocumentNoise, seeded_rngs
from signature_training.paths import DEFAULT_FONT


@pytest.fixture
def blank():
    return np.full((512, 512, 3), 255, dtype=np.uint8)


@pytest.fixture
def noise():
    return DocumentNoise(DEFAULT_FONT, rng=np.random.default_rng(0))


# ── the regression that motivated this module ─────────────────────────────────


def test_horizontal_rules_differ_between_images(blank, noise):
    """The old add_random_straight_lines built
    np.random.default_rng(seed=21520063) on every call, so 1000 calls produced
    exactly one distinct output: always two lines, at y=265 and y=521.
    """
    outputs = [noise.horizontal_rules(blank.copy()) for _ in range(25)]
    unique = {o.tobytes() for o in outputs}
    assert len(unique) > 15, (
        f"only {len(unique)} distinct results from 25 calls - the RNG is being "
        "re-seeded per call again"
    )


def test_line_positions_actually_vary(blank, noise):
    """Distinctness alone could come from thickness; check the y positions."""
    rows = set()
    for _ in range(25):
        out = noise.horizontal_rules(blank.copy())
        dark = np.where((out[:, :, 0] < 128).sum(axis=1) > 256)[0]
        rows.update(dark.tolist())
    assert len(rows) > 10


def test_same_seed_reproduces_the_same_sequence(blank):
    def run():
        n = DocumentNoise(DEFAULT_FONT, rng=np.random.default_rng(1234))
        return [n(blank.copy()).tobytes() for _ in range(5)]

    assert run() == run()


def test_different_seeds_diverge(blank):
    def run(seed):
        n = DocumentNoise(DEFAULT_FONT, rng=np.random.default_rng(seed))
        return [n(blank.copy()).tobytes() for _ in range(5)]

    assert run(1) != run(2)


# ── font resolution ───────────────────────────────────────────────────────────


def test_bundled_font_exists():
    """The old code looked for 'cyclegan_unprocessed_data/times.ttf' relative to
    the process cwd, which resolved under no documented working directory."""
    assert DEFAULT_FONT.is_file()


def test_missing_font_fails_immediately_not_per_image(tmp_path):
    """A missing font must stop the run at construction. Previously the OSError
    surfaced once per image inside a broad except, and the build reported
    success having written nothing."""
    with pytest.raises(FileNotFoundError):
        DocumentNoise(tmp_path / "nope.ttf", rng=np.random.default_rng(0))


# ── effects do something ──────────────────────────────────────────────────────


def test_caption_draws_dark_pixels(blank, noise):
    out = noise.caption(blank.copy())
    assert (out < 128).any(), "caption drew nothing"


def test_cell_borders_are_near_the_edges(blank):
    n = DocumentNoise(DEFAULT_FONT, rng=np.random.default_rng(7))
    columns = set()
    for _ in range(40):
        out = n.cell_borders(blank.copy())
        dark = np.where((out[:, :, 0] < 128).sum(axis=0) > 256)[0]
        columns.update(dark.tolist())
    if columns:
        margin = int(0.15 * 512) + 4
        assert all(c <= margin or c >= 512 - margin for c in columns)


def test_full_pipeline_changes_the_image(blank, noise):
    assert not np.array_equal(noise(blank.copy()), blank)


def test_probabilities_of_zero_leave_the_image_untouched(blank):
    n = DocumentNoise(DEFAULT_FONT, rng=np.random.default_rng(0), p_lines=0.0, p_text=0.0)
    assert np.array_equal(n(blank.copy()), blank)


def test_seeded_rngs_are_independent_and_reproducible():
    np_a, py_a = seeded_rngs(99)
    np_b, py_b = seeded_rngs(99)
    assert np_a.random() == np_b.random()
    assert py_a.random() == py_b.random()
