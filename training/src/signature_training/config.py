"""Pipeline configuration.

One YAML file describes every stage, so the stages agree on paths instead of
each taking its own `--src`/`--dst` and drifting. CLI flags override the file;
the file overrides the defaults here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import DEFAULT_FONT, PROJECT_ROOT


@dataclass
class PathsConfig:
    """Everything the pipeline reads or writes.

    Relative paths resolve against the training/ directory, not the caller's
    working directory, so a stage behaves the same however it is invoked.
    """

    raw_signatures: str = "data/raw/sign_data"
    cyclegan_dataset: str = "data/processed/cyclegan"
    verification_dataset: str = "data/processed/verification"
    stamps: str = "data/raw/stamps"
    cyclegan_repo: str = "external/pytorch-CycleGAN-and-pix2pix"
    cyclegan_checkpoints: str = "artifacts/cyclegan"
    models: str = "artifacts/models"
    evaluation: str = "artifacts/evaluation"
    onnx: str = "artifacts/onnx"
    triton_repository: str = "../inference/triton/model_repository"
    font: str = str(DEFAULT_FONT)

    def resolve(self, name: str) -> Path:
        value = Path(getattr(self, name))
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


@dataclass
class ColourConfig:
    """Paper white-balance, for the whole pipeline.

    ONE field on purpose. The Kaggle scans are ~8 levels short on red in the
    paper (R 243.28 against G/B 251.47), which reads as a cyan cast; CycleGAN
    reproduces whatever colour distribution it is trained on, so the denoiser
    emits the cast too (+3.45). The verification backbones were then trained on
    raw scans at +8.18 and served denoised crops at +3.45 — two different
    distributions, neither normalised.

    `mode` is applied wherever a dataset is WRITTEN, so everything downstream
    reads one paper white point:

        data/cyclegan.py:_make_pair     both domains of the CycleGAN pair set
        data/cyclegan.py:_copy_person   the images the backbones read

    Once, at build time — not in the Keras hook, which runs after augmentation
    and would compose two clamped gains. See models/preprocess.py.

    The numeric parameters — paper level, max gain, strength — are deliberately
    NOT here. They live as module constants in data/colour.py, which is
    byte-identical to the inference service's copy and kept that way by CI. That
    leaves exactly one string that can disagree between the two halves instead of
    six.

    Changing `mode` invalidates the built datasets AND the trained extractors.
    The builders stamp each dataset with the mode it was written under and refuse
    to extend it under another, so a change means deleting
    data/processed/verification/ and re-running data-verification,
    train-verification, evaluate, export.
    """

    # "none" | "whiten" | "desaturate"
    mode: str = "none"


@dataclass
class CycleGANDataConfig:
    image_size: int = 512
    test_ratio: float = 0.10
    seed: int = 42
    # Probability a given clean image also receives each noise type.
    p_lines: float = 0.9
    p_text: float = 0.9
    p_stamp: float = 0.70


@dataclass
class CycleGANTrainConfig:
    name: str = "signature"
    n_epochs: int = 100
    n_epochs_decay: int = 100
    load_size: int = 286
    crop_size: int = 256
    norm: str = "instance"
    gpu_ids: str = "0"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class VerificationTrainConfig:
    backbone: str = "both"  # vgg16 | resnet50 | both
    image_size: int = 224
    batch_size: int = 32
    val_split: float = 0.15
    phase1_epochs: int = 50  # frozen backbone, head only
    phase2_epochs: int = 10  # full fine-tune
    phase1_lr: float = 1e-3
    phase2_lr: float = 1e-4
    embedding_dim: int = 4096
    seed: int = 42
    vgg16_weights: str = ""  # optional local .h5, else Keras downloads
    resnet50_weights: str = ""


@dataclass
class EvaluateConfig:
    image_size: int = 224
    # Cap on impostor pairs sampled per person-pair. The default of 1 gives a
    # coarse ROC in the low-FAR region; raise it to resolve TAR @ FAR 0.1%.
    impostor_pairs_per_couple: int = 8
    seed: int = 42


@dataclass
class ExportConfig:
    opset: int = 13
    image_size: int = 224
    yolo_image_size: int = 640
    max_batch_size: int = 1
    instance_kind: str = "KIND_GPU"  # KIND_GPU | KIND_CPU
    simplify: bool = True


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    colour: ColourConfig = field(default_factory=ColourConfig)
    cyclegan_data: CycleGANDataConfig = field(default_factory=CycleGANDataConfig)
    cyclegan_train: CycleGANTrainConfig = field(default_factory=CycleGANTrainConfig)
    verification: VerificationTrainConfig = field(default_factory=VerificationTrainConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        data: dict[str, Any] = {}
        if path is not None:
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}

        cfg = cls(
            paths=_build(PathsConfig, data.get("paths")),
            colour=_build(ColourConfig, data.get("colour")),
            cyclegan_data=_build(CycleGANDataConfig, data.get("cyclegan_data")),
            cyclegan_train=_build(CycleGANTrainConfig, data.get("cyclegan_train")),
            verification=_build(VerificationTrainConfig, data.get("verification")),
            evaluate=_build(EvaluateConfig, data.get("evaluate")),
            export=_build(ExportConfig, data.get("export")),
        )
        for dotted, value in (overrides or {}).items():
            cfg.set(dotted, value)
        return cfg

    def set(self, dotted: str, value: Any) -> None:
        """Apply a `section.field=value` override, coercing to the field type."""
        section, _, name = dotted.partition(".")
        if not name:
            raise ValueError(f"Override must be section.field=value, got {dotted!r}")
        if not hasattr(self, section):
            raise ValueError(f"Unknown config section {section!r}")
        target = getattr(self, section)
        if not hasattr(target, name):
            raise ValueError(f"Unknown field {name!r} in section {section!r}")

        current_type = type(getattr(target, name))
        if isinstance(value, str) and current_type is not str:
            if current_type is bool:
                value = value.strip().lower() in {"1", "true", "yes", "on"}
            elif current_type is list:
                value = [v for v in value.split(",") if v]
            else:
                value = current_type(value)
        setattr(target, name, value)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str | Path) -> None:
        """Write the fully resolved config next to the run's outputs, so a
        result can always be traced back to the settings that produced it."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


# Keys that used to exist, and where they went. A config file written against an
# older revision then fails with the new name rather than with a bare "unknown
# key", which is the difference between a one-line edit and an afternoon.
_MOVED = {
    ("CycleGANDataConfig", "colour_mode"): (
        "colour.mode — the setting now governs the extractor input as well as "
        "the CycleGAN pair set, so it is no longer specific to cyclegan_data"
    ),
}


def _build(cls, data: dict[str, Any] | None):
    if not data:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        moved = [
            f"  {key!r} moved to {_MOVED[(cls.__name__, key)]}"
            for key in sorted(unknown)
            if (cls.__name__, key) in _MOVED
        ]
        raise ValueError(
            "\n".join(
                [
                    f"Unknown key(s) {sorted(unknown)} in config section {cls.__name__}.",
                    *moved,
                    f"Valid keys: {sorted(known)}",
                ]
            )
        )
    return cls(**data)
