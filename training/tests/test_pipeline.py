"""Dataset building, pair construction, Triton config generation, and the CLI.

Everything here runs without TensorFlow, torch or a GPU.
"""

import numpy as np
import pytest
from PIL import Image

from signature_training.config import Config
from signature_training.data import cyclegan
from signature_training.evaluate import pairs


# ── dataset builder ───────────────────────────────────────────────────────────


@pytest.fixture
def corpus(tmp_path):
    """A miniature Kaggle-shaped tree: 3 genuine people + their forgeries."""
    root = tmp_path / "raw" / "sign_data"
    for split, people in (("train", ["001", "002", "003"]), ("test", ["004", "005"])):
        for person in people:
            for suffix in ("", "_forg"):
                d = root / split / f"{person}{suffix}"
                d.mkdir(parents=True)
                for i in range(3):
                    Image.new("RGB", (220, 90), (255, 255, 255)).save(d / f"{person}_{i}.png")
    return root


@pytest.fixture
def cfg(tmp_path, corpus):
    return Config.load(overrides={
        "paths.raw_signatures": str(corpus),
        "paths.cyclegan_dataset": str(tmp_path / "processed" / "cyclegan"),
        "paths.verification_dataset": str(tmp_path / "processed" / "verification"),
        "paths.stamps": str(tmp_path / "no_stamps"),
        "cyclegan_data.image_size": "64",
    })


def test_build_writes_paired_domains(cfg):
    result = cyclegan.build(cfg)
    dst = cfg.paths.resolve("cyclegan_dataset")

    assert result["failed"] == 0
    assert result["train"] + result["test"] > 0
    for split in ("train", "test"):
        a = sorted(p.name for p in (dst / f"{split}A").glob("*.png"))
        b = sorted(p.name for p in (dst / f"{split}B").glob("*.png"))
        assert a == b, "A and B must be paired by filename"


def test_noisy_domain_actually_differs_from_clean(cfg):
    cyclegan.build(cfg)
    dst = cfg.paths.resolve("cyclegan_dataset")
    differing = 0
    for a in (dst / "trainA").glob("*.png"):
        b = dst / "trainB" / a.name
        if not np.array_equal(np.array(Image.open(a)), np.array(Image.open(b))):
            differing += 1
    assert differing > 0, "domain B is identical to domain A - no noise applied"


def test_build_is_reproducible_under_a_fixed_seed(cfg, tmp_path):
    cyclegan.build(cfg)
    first = {p.name: p.read_bytes()
             for p in (cfg.paths.resolve("cyclegan_dataset") / "trainB").glob("*.png")}

    cfg.paths.cyclegan_dataset = str(tmp_path / "again")
    cyclegan.build(cfg)
    second = {p.name: p.read_bytes()
              for p in (cfg.paths.resolve("cyclegan_dataset") / "trainB").glob("*.png")}

    assert first.keys() == second.keys()
    assert first == second


def test_empty_source_raises_rather_than_reporting_success(cfg, tmp_path):
    cfg.paths.raw_signatures = str(tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No images"):
        cyclegan.build(cfg)


def test_make_square_pads_white_and_preserves_aspect():
    wide = Image.new("RGB", (400, 100), (0, 0, 0))
    out = cyclegan.make_square(wide, 128)
    assert out.size == (128, 128)
    arr = np.array(out)
    assert arr[:4, :4].mean() > 200, "corners should be white padding"


# ── verification split ────────────────────────────────────────────────────────


def test_verification_split_excludes_forgeries(cfg):
    counts = cyclegan.build_verification_split(cfg)
    dst = cfg.paths.resolve("verification_dataset")

    assert counts["train"] == 3 and counts["test"] == 2
    for split in ("train", "test"):
        names = [p.name for p in (dst / split).iterdir() if p.is_dir()]
        assert names, f"{split} is empty"
        assert not any("forg" in n for n in names), f"forgeries leaked into {split}"


# ── pair building ─────────────────────────────────────────────────────────────


def _fake_embedder(dim=32):
    """Embedding determined by the person id, so same-person pairs are similar
    and different-person pairs are not."""
    def embed(path):
        person = path.parent.name.replace("_org", "")
        rng = np.random.default_rng(abs(hash(person)) % (2**31))
        base = rng.normal(size=dim)
        jitter = np.random.default_rng(abs(hash(path.name)) % (2**31)).normal(scale=0.05, size=dim)
        vec = base + jitter
        return vec / np.linalg.norm(vec)
    return embed


@pytest.fixture
def test_people(tmp_path):
    root = tmp_path / "test"
    for person in ("001", "002", "003", "004"):
        d = root / person
        d.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (60, 30), (255, 255, 255)).save(d / f"{person}_{i}.png")
    (root / "005_forg").mkdir()
    return root


def test_pairs_include_both_classes_and_skip_forgeries(test_people):
    scores, labels = pairs.build(test_people, _fake_embedder(), impostor_pairs_per_couple=4)
    assert set(labels) == {0, 1}
    # 4 people x C(4,2) within-person pairs
    assert (labels == 1).sum() == 4 * 6


def test_impostor_sampling_scales_with_the_setting(test_people):
    _, few = pairs.build(test_people, _fake_embedder(), impostor_pairs_per_couple=1)
    _, many = pairs.build(test_people, _fake_embedder(), impostor_pairs_per_couple=8)
    assert (many == 0).sum() > (few == 0).sum()


def test_genuine_pairs_score_higher_than_impostor_pairs(test_people):
    scores, labels = pairs.build(test_people, _fake_embedder(), impostor_pairs_per_couple=4)
    assert scores[labels == 1].mean() > scores[labels == 0].mean()


def test_single_person_is_rejected(tmp_path):
    root = tmp_path / "one"
    d = root / "001"
    d.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (60, 30)).save(d / f"a{i}.png")
    with pytest.raises(ValueError, match="genuine and impostor"):
        pairs.build(root, _fake_embedder())


def test_missing_test_dir_is_reported_clearly(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No person subfolders"):
        pairs.person_dirs(tmp_path / "empty")


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_help_exits_cleanly():
    from signature_training.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_rejects_a_malformed_override():
    from signature_training.cli import main

    assert main(["--set", "novalue", "export"]) == 2


def test_cli_rejects_a_missing_config():
    from signature_training.cli import main

    assert main(["--config", "/nonexistent.yaml", "export"]) == 2


def test_cli_exposes_every_stage():
    from signature_training.cli import STAGE_ORDER, STAGES

    assert set(STAGE_ORDER) <= set(STAGES)
    assert "setup" in STAGES
