"""Printed-document noise: form rules, cell borders, and caption text.

Extracted from the old dataset_preparation.py, with two defects fixed.

1. RNG lifetime. `add_random_straight_lines` built
   `np.random.default_rng(seed=21520063)` *inside* the function, so it was
   re-seeded identically on every call. All N noisy images received the same
   two lines at the same two y positions, and `--seed` had no effect on them.
   Verified: 1000 calls produced exactly one distinct result. The generator
   could then learn to erase those two rows rather than to remove lines.
   The RNG is now owned by the caller and threaded through.

2. Font resolution. The font was loaded from the relative path
   "cyclegan_unprocessed_data/times.ttf", which resolved against the process
   cwd and matched nothing under any documented working directory. The
   resulting OSError was swallowed by a broad `except Exception` in the
   per-image loop, so a run could report success having written no pairs at
   all. The font now comes from the package's assets directory and is opened
   once, up front, so a missing font fails immediately.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAMES = [
    "Nguyễn Thị B",
    "Đỗ Văn C",
    "Phạm Văn D",
    "Trần Mỹ Thoa",
    "Trịnh Thị Dung",
    "Tống Thành Tuấn",
    "Amal Joseph",
    "Steve Jobs",
    "Larry Page",
    "Katie Bouman",
    "Ada Lovelace",
]

CAPTIONS = [
    "(Ký, họ tên)",
    "(Ký, họ tên, đóng dấu)",
    "Sincerely,",
    "Regards,",
    "Yours truly,",
]


class DocumentNoise:
    """Adds the printed furniture that surrounds a signature on a real form.

    Parameters
    ----------
    font_path : the TrueType font used for caption text. Opened eagerly.
    rng       : numpy Generator owned by the caller, so a single seed governs
                the whole dataset build and successive images differ.
    """

    def __init__(
        self,
        font_path: str | Path,
        rng: np.random.Generator,
        p_lines: float = 0.9,
        p_text: float = 0.9,
    ) -> None:
        self.font_path = str(font_path)
        self.rng = rng
        self.p_lines = p_lines
        self.p_text = p_text

        # Fail now, loudly, rather than once per image inside a try/except.
        if not Path(self.font_path).is_file():
            raise FileNotFoundError(f"Caption font not found: {self.font_path}")
        ImageFont.truetype(self.font_path, 20)

    # ── individual effects ────────────────────────────────────────────────────

    def horizontal_rules(self, image: np.ndarray) -> np.ndarray:
        """1-2 horizontal printed form lines."""
        height, width = image.shape[:2]
        num_lines = int(self.rng.integers(1, 3))
        y0 = max(1, height // num_lines)
        for i in range(num_lines):
            thickness = int(self.rng.integers(1, 4))
            jitter = int(self.rng.uniform(-0.05 * height, 0.05 * height))
            y = int(np.clip(y0 * (i + 1) + jitter, 0, height - 1))
            image = cv2.line(image, (0, y), (width, y), (0, 0, 0), thickness=thickness)
        return image

    def cell_borders(self, image: np.ndarray) -> np.ndarray:
        """0-2 vertical borders of a signature box, placed near the edges.

        Real crops show no border (~50%), one (~35%) or both (~15%).
        """
        height, width = image.shape[:2]
        num_lines = int(self.rng.choice([0, 1, 2], p=[0.50, 0.35, 0.15]))
        if num_lines == 0:
            return image

        edge_margin = max(1, int(0.15 * width))
        sides = (
            ["left", "right"]
            if num_lines == 2
            else ["left" if self.rng.random() < 0.5 else "right"]
        )

        for side in sides:
            thickness = int(self.rng.integers(1, 4))
            if side == "left":
                x = int(self.rng.integers(0, edge_margin))
            else:
                x = int(self.rng.integers(max(0, width - edge_margin), width))
            y_start = int(self.rng.integers(0, max(1, int(0.10 * height))))
            y_end = int(self.rng.integers(int(0.90 * height), height))
            image = cv2.line(image, (x, y_start), (x, y_end), (0, 0, 0), thickness=thickness)
        return image

    def caption(self, image: np.ndarray) -> np.ndarray:
        """A '(Ký, họ tên)' style caption with a printed name below it."""
        height, width = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)

        label_y = int(np.clip(self.rng.uniform(0.60 * height, 0.75 * height), 0, height - 1))
        label_x = int(np.clip(self.rng.uniform(0.05 * width, 0.25 * width), 0, width - 1))
        label_font = ImageFont.truetype(
            self.font_path, max(8, int(self.rng.uniform(0.07, 0.10) * height))
        )
        draw.text(
            (label_x, label_y),
            CAPTIONS[int(self.rng.integers(len(CAPTIONS)))],
            font=label_font,
            fill=(0, 0, 0),
        )

        name_y = int(np.clip(label_y + self.rng.integers(30, 50), 0, height - 1))
        name_font = ImageFont.truetype(
            self.font_path, max(8, int(self.rng.uniform(0.12, 0.15) * height))
        )
        draw.text(
            (label_x, name_y),
            NAMES[int(self.rng.integers(len(NAMES)))],
            font=name_font,
            fill=(0, 0, 0),
        )

        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ── combined ──────────────────────────────────────────────────────────────

    def __call__(self, image: np.ndarray) -> np.ndarray:
        out = image
        if self.rng.random() < self.p_lines:
            out = self.horizontal_rules(out)
            out = self.cell_borders(out)
        if self.rng.random() < self.p_text:
            out = self.caption(out)
        return out


def seeded_rngs(seed: int) -> tuple[np.random.Generator, random.Random]:
    """One seed, two generators - numpy for the document noise, stdlib for the
    stamp augmentor, which uses `random` throughout."""
    return np.random.default_rng(seed), random.Random(seed)
