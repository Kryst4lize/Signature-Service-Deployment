"""Package-relative asset and root resolution.

Every path here is derived from this file's own location, never from the
process working directory.

That is a deliberate fix, not a style choice: the old
`font_path = "cyclegan_unprocessed_data/times.ttf"` in dataset_preparation.py
resolved against `os.getcwd()`, so it never found the font under any documented
working directory. `ImageFont.truetype` raised OSError, the caller's broad
`except Exception` swallowed it per image, and the run finished by reporting a
dataset it had in fact written zero pairs into.
"""

from pathlib import Path

# .../training/src/signature_training/paths.py -> .../training
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"

DEFAULT_FONT = ASSETS_DIR / "times.ttf"


def asset(name: str) -> Path:
    """Resolve a bundled asset, failing loudly if it is missing.

    Raising here is the point: a missing font must stop the run, not silently
    degrade every generated sample.
    """
    path = ASSETS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled asset {name!r} not found at {path}. "
            "The package layout is broken or the file was not installed."
        )
    return path
