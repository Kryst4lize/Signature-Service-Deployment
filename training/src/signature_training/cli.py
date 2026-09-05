"""`sigtrain` - the end-to-end training pipeline driver.

    sigtrain setup                clone the CycleGAN repo, check the data layout
    sigtrain data-cyclegan        build paired clean/noisy images
    sigtrain data-verification    filter genuine-only person folders
    sigtrain train-cyclegan       train the denoiser
    sigtrain train-verification   fine-tune VGG16 + ResNet50, save extractors
    sigtrain evaluate             EER / AUC / d' on held-out identities
    sigtrain export               ONNX + config.pbtxt into the Triton repository
    sigtrain all                  every stage above, in order

Heavy imports (TensorFlow, torch, ultralytics) happen inside the stage that
needs them, so `sigtrain --help` and the data stages stay fast and do not
require a deep-learning stack to be installed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .paths import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

STAGE_ORDER = [
    "data-cyclegan",
    "data-verification",
    "train-cyclegan",
    "train-verification",
    "evaluate",
    "export",
]


def _log_setup(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _banner(stage: str) -> None:
    print(f"\n{'=' * 70}\n  {stage}\n{'=' * 70}")


# ── stages ────────────────────────────────────────────────────────────────────


def stage_setup(cfg: Config, args) -> None:
    from .train import cyclegan as cyclegan_train

    repo = cyclegan_train.ensure_repo(cfg)
    print(f"CycleGAN repo ready at {repo}")

    raw = cfg.paths.resolve("raw_signatures")
    if raw.is_dir():
        people = sum(1 for _ in raw.rglob("*") if _.is_dir())
        print(f"Raw dataset at {raw} ({people} subdirectories)")
    else:
        print(
            f"\nRaw dataset NOT found at {raw}\n"
            "  Download the Kaggle signature-verification-dataset and extract it\n"
            "  so that the tree looks like:\n"
            f"    {raw}/train/001/, 001_forg/, 002/, ...\n"
            f"    {raw}/test/...\n"
            "  https://www.kaggle.com/datasets/robinreni/signature-verification-dataset"
        )

    stamps = cfg.paths.resolve("stamps")
    if not stamps.is_dir() or not any(stamps.iterdir()):
        print(
            f"\nNo stamp images at {stamps}\n"
            "  Stamp noise will be disabled, and the denoiser will not learn to\n"
            "  remove seals. Add a few dozen stamp/seal images to enable it."
        )


def stage_data_cyclegan(cfg: Config, args) -> None:
    from .data import cyclegan

    result = cyclegan.build(cfg)
    print(f"CycleGAN dataset: {result}")


def stage_data_verification(cfg: Config, args) -> None:
    from .data import cyclegan

    result = cyclegan.build_verification_split(cfg)
    print(f"Verification dataset: {result}")


def stage_train_cyclegan(cfg: Config, args) -> None:
    from .train import cyclegan as cyclegan_train

    out = cyclegan_train.run(cfg, resume_epoch=getattr(args, "resume_epoch", None))
    print(f"CycleGAN checkpoints: {out}")


def stage_train_verification(cfg: Config, args) -> None:
    from .train import verification

    produced = verification.run(cfg)
    for backbone, path in produced.items():
        print(f"  {backbone}: {path}")


def stage_evaluate(cfg: Config, args) -> None:
    from .evaluate import runner

    results = runner.run(cfg)
    print("\nSet in inference/.env:")
    for r in results:
        print(f"  # {r['name']}: EER {r['eer']:.4f}")
    best = min(results, key=lambda r: r["eer"])
    print(f"  MATCH_THRESHOLD={best['match_threshold_for_service']:.4f}   # from {best['name']}")


def stage_export(cfg: Config, args) -> None:
    from .export import repository

    produced = repository.build(cfg, only=getattr(args, "only", None))
    print(f"\nTriton model repository at {cfg.paths.resolve('triton_repository')}:")
    for name in produced:
        print(f"  {name}/1/model.onnx  +  config.pbtxt")
    print("\nStart the service with:  cd ../inference && docker compose up -d")


STAGES = {
    "setup": stage_setup,
    "data-cyclegan": stage_data_cyclegan,
    "data-verification": stage_data_verification,
    "train-cyclegan": stage_train_cyclegan,
    "train-verification": stage_train_verification,
    "evaluate": stage_evaluate,
    "export": stage_export,
}


def stage_all(cfg: Config, args) -> None:
    skip = set(getattr(args, "skip", None) or [])
    for stage in STAGE_ORDER:
        if stage in skip:
            print(f"\n[skipping {stage}]")
            continue
        _banner(stage)
        STAGES[stage](cfg, args)


# ── entry point ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    # The global flags go on a parent parser that is attached to BOTH the root
    # and every subparser, so `sigtrain --set k=v export` and
    # `sigtrain export --set k=v` both work. With them on the root alone,
    # argparse hands everything after the stage name to the subparser, which
    # does not know `--set` and exits 2 with "unrecognized arguments" — and the
    # post-stage order is the one every doc and example uses.
    #
    # The defaults are SUPPRESS so the subparser does not overwrite a value the
    # root already captured; real defaults are applied after parsing.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", default=argparse.SUPPRESS, help=f"YAML config (default: {DEFAULT_CONFIG.name})"
    )
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=argparse.SUPPRESS,
        metavar="section.field=value",
        help="Override a config value; repeatable",
    )
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(
        prog="sigtrain",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )

    sub = parser.add_subparsers(dest="stage", required=True)
    for name in STAGES:
        sp = sub.add_parser(name, parents=[common], help=STAGES[name].__doc__ or name)
        if name == "train-cyclegan":
            sp.add_argument(
                "--resume-epoch",
                type=int,
                default=None,
                help="Resume from this epoch (passes --continue_train)",
            )
        if name == "export":
            sp.add_argument(
                "--only",
                nargs="+",
                default=None,
                metavar="MODEL",
                help="Export only these model names",
            )
            sp.add_argument(
                "--output", default=None, help="Triton repository path (overrides the config)"
            )

    sp_all = sub.add_parser("all", parents=[common], help="run every stage in order")
    sp_all.add_argument("--skip", nargs="+", default=[], choices=STAGE_ORDER, help="Stages to skip")
    sp_all.add_argument("--resume-epoch", type=int, default=None)
    sp_all.add_argument("--only", nargs="+", default=None)
    sp_all.add_argument("--output", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # SUPPRESS defaults mean these may be absent entirely.
    args.config = getattr(args, "config", str(DEFAULT_CONFIG))
    args.overrides = getattr(args, "overrides", []) or []
    args.verbose = getattr(args, "verbose", False)

    _log_setup(args.verbose)

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            print(f"error: --set expects section.field=value, got {item!r}", file=sys.stderr)
            return 2
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 2

    try:
        cfg = Config.load(config_path, overrides)
        if getattr(args, "output", None):
            cfg.paths.triton_repository = args.output
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    handler = stage_all if args.stage == "all" else STAGES[args.stage]
    try:
        if args.stage != "all":
            _banner(args.stage)
        handler(cfg, args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
