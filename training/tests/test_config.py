"""Config loading, overriding and path resolution."""

import pytest
import yaml

from signature_training.config import Config
from signature_training.paths import PROJECT_ROOT


def test_defaults_load_without_a_file():
    cfg = Config.load()
    assert cfg.verification.backbone == "both"
    assert cfg.verification.embedding_dim == 4096
    assert cfg.export.opset == 13


def test_shipped_default_yaml_is_valid():
    """Guards against the config drifting away from the dataclasses: an unknown
    key raises rather than being silently ignored."""
    cfg = Config.load(PROJECT_ROOT / "configs" / "default.yaml")
    assert cfg.cyclegan_train.crop_size == 256
    assert cfg.paths.triton_repository.endswith("triton/model_repository")


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"verification": {"backbne": "vgg16"}}))
    with pytest.raises(ValueError, match="Unknown key"):
        Config.load(path)


def test_file_overrides_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"verification": {"batch_size": 8}}))
    cfg = Config.load(path)
    assert cfg.verification.batch_size == 8
    assert cfg.verification.val_split == 0.15  # untouched default


def test_cli_override_beats_the_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"verification": {"batch_size": 8}}))
    cfg = Config.load(path, {"verification.batch_size": "64"})
    assert cfg.verification.batch_size == 64


@pytest.mark.parametrize(
    "dotted,value,expected",
    [
        ("verification.batch_size", "16", 16),
        ("evaluate.image_size", "128", 128),
        ("cyclegan_data.test_ratio", "0.25", 0.25),
        ("export.simplify", "false", False),
        ("export.simplify", "yes", True),
        ("cyclegan_train.extra_args", "--a,--b", ["--a", "--b"]),
    ],
)
def test_overrides_coerce_to_the_field_type(dotted, value, expected):
    cfg = Config.load(overrides={dotted: value})
    section, _, name = dotted.partition(".")
    assert getattr(getattr(cfg, section), name) == expected


def test_unknown_override_target_is_rejected():
    with pytest.raises(ValueError, match="Unknown config section"):
        Config.load(overrides={"nope.field": "1"})
    with pytest.raises(ValueError, match="Unknown field"):
        Config.load(overrides={"verification.nope": "1"})


def test_malformed_override_is_rejected():
    with pytest.raises(ValueError, match="section.field"):
        Config.load(overrides={"batch_size": "8"})


# ── path resolution ───────────────────────────────────────────────────────────


def test_relative_paths_resolve_against_the_project_not_the_cwd(monkeypatch, tmp_path):
    """The whole point of PathsConfig.resolve: a stage must behave identically
    however it was invoked."""
    cfg = Config.load()
    monkeypatch.chdir(tmp_path)
    assert cfg.paths.resolve("models") == (PROJECT_ROOT / "artifacts/models").resolve()


def test_absolute_paths_are_left_alone():
    cfg = Config.load(overrides={"paths.models": "/srv/models"})
    assert str(cfg.paths.resolve("models")) == "/srv/models"


def test_triton_repository_points_into_the_inference_service():
    cfg = Config.load(PROJECT_ROOT / "configs" / "default.yaml")
    resolved = cfg.paths.resolve("triton_repository")
    assert resolved == (PROJECT_ROOT.parent / "inference/triton/model_repository").resolve()


def test_dump_round_trips(tmp_path):
    cfg = Config.load(overrides={"verification.batch_size": "11"})
    out = tmp_path / "used.yaml"
    cfg.dump(out)
    assert Config.load(out).verification.batch_size == 11
