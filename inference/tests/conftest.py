import os

# Settings are constructed at import time, so the environment has to be
# populated before anything under app/ is imported.
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("TRITON_HOST", "localhost")

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rgb_image() -> Image.Image:
    """A 300x120 image with three unambiguous colour blocks, so channel-order
    mistakes show up as wrong numbers rather than as a plausible-looking image.
    """
    arr = np.zeros((120, 300, 3), dtype=np.uint8)
    arr[:, :100] = (255, 0, 0)      # red
    arr[:, 100:200] = (0, 255, 0)   # green
    arr[:, 200:] = (0, 0, 255)      # blue
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def unit_tensor() -> np.ndarray:
    """[1, 3, 8, 8] float32 spanning the full [0, 1] range."""
    rng = np.random.default_rng(0)
    return rng.random((1, 3, 8, 8), dtype=np.float32)
