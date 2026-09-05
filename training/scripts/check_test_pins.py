"""Fail if tests/Dockerfile's pins have drifted from uv.lock.

The test runner deliberately installs a hand-picked subset rather than running
`uv sync`: the full locked set is ~9 GB of CUDA wheels, which would make
`make test` unusable as a feedback loop.

That duplication is the price, and this is what stops it rotting — a version
bumped in pyproject.toml but not here would otherwise mean the tests silently
run against different libraries than the pipeline does.

    uv run python scripts/check_test_pins.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "uv.lock"
DOCKERFILE = ROOT / "tests" / "Dockerfile"

# Installed by the test image but not a runtime dependency of the project.
TEST_ONLY = {"pytest"}


def locked_versions() -> dict[str, str]:
    data = tomllib.loads(LOCK.read_text())
    return {p["name"].lower(): p["version"] for p in data["package"]}


def dockerfile_pins() -> dict[str, str]:
    pins = {}
    for name, version in re.findall(
        r"^\s+([A-Za-z0-9._-]+)==([A-Za-z0-9.]+)\s*\\?\s*$",
        DOCKERFILE.read_text(),
        re.MULTILINE,
    ):
        pins[name.lower().replace("_", "-")] = version
    return pins


def main() -> int:
    locked = locked_versions()
    pinned = dockerfile_pins()

    if not pinned:
        print("error: found no pins in tests/Dockerfile - has its format changed?")
        return 1

    problems = []
    for name, version in sorted(pinned.items()):
        if name in TEST_ONLY:
            continue
        if name not in locked:
            problems.append(
                f"  {name}=={version} is pinned in tests/Dockerfile but absent from uv.lock"
            )
        elif locked[name] != version:
            problems.append(f"  {name}: tests/Dockerfile has {version}, uv.lock has {locked[name]}")

    if problems:
        print("tests/Dockerfile has drifted from uv.lock:")
        print("\n".join(problems))
        print("\nUpdate tests/Dockerfile to match, or run `make lock` if pyproject changed.")
        return 1

    print(f"tests/Dockerfile pins agree with uv.lock ({len(pinned)} checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
