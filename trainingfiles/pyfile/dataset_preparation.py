"""
dataset_preparation.py
─────────────────────────────────────────────────────────────────────────────
Prepares a paired (clean A ↔ noisy B) dataset for CycleGAN training.

Pipeline
────────
  1. Collect all clean signature images from the source folder.
  2. For every image generate a noisy counterpart by applying:
       • Random horizontal lines   (printed form lines)
       • Random printed text       (name / title text below signature)
       • Stamp / seal overlay      (DPI-aware, Vietnamese placement rules)
  3. Resize both domains to 512×512 (signature centred, white padding).
  4. Split into trainA / trainB / testA / testB folders.

Usage
─────
    python dataset_preparation.py \
        --src      data/clean_signatures \
        --dst      data/cyclegan_dataset \
        --stamps   stamp_noise \
        --split    0.1

Directory layout produced
──────────────────────────
    data/cyclegan_dataset/
        trainA/   ← clean signatures
        trainB/   ← noisy counterparts
        testA/
        testB/
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from stamp_augmentation import StampAugmentor

# ─────────────────────────────────────────────────────────────────────────────
# Noise functions
# ─────────────────────────────────────────────────────────────────────────────

def add_random_straight_lines(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Add 1-4 horizontal printed-form lines across the image."""
    rng = np.random.default_rng(seed=21520063)
    num_lines = rng.integers(1, 3)
    y0 = max(1, height // num_lines)
    for i in range(num_lines):
        thickness = int(rng.integers(1, 4))
        jitter = int(rng.uniform(-0.05 * height, 0.05 * height))
        y = y0 * (i + 1) + jitter
        y = int(np.clip(y, 0, height - 1))
        image = cv2.line(image, (0, y), (width, y), (0, 0, 0), thickness=thickness)
    return image

def add_random_text(image: np.ndarray, height: int, width: int) -> np.ndarray:
    names = [
        "Nguyễn Thị B", "Đỗ Văn C", "Phạm Văn D",
        "Trần Mỹ Thoa", "Trịnh Thị Dung", "Tống Thành Tuấn",
        "Amal Joseph",   "Steve Jobs",     "Larry Page",
        "Katie Bouman",  "Ada Lovelace",
    ]
    labels = [
        "(Ký, họ tên)", "(Ký, họ tên, đóng dấu)",
        "Sincerely,", "Regards,", "Yours truly,",
    ]

    # Chuyển sang PIL để vẽ được tiếng Việt
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Đường dẫn font (Thay bằng đường dẫn font trên máy bạn, ví dụ 'arial.ttf')
    font_path = "cyclegan_unprocessed_data/times.ttf" 

    # 1. Vẽ Label (Ký tên)
    label_y = int(np.clip(np.random.uniform(0.60 * height, 0.75 * height), 0, height - 1))
    label_x = int(np.clip(np.random.uniform(0.05 * width, 0.25 * width), 0, width - 1))
    
    label_scale = np.random.uniform(0.07, 0.1) # Đã tăng để to hơn theo ý bạn
    font_label = ImageFont.truetype(font_path, int(label_scale * height))
    
    draw.text((label_x, label_y), random.choice(labels), font=font_label, fill=(0, 0, 0))

    # 2. Vẽ Name (Tên bên dưới)
    name_y = int(np.clip(label_y + np.random.randint(30, 50), 0, height - 1))
    name_x = label_x
    
    name_scale = np.random.uniform(0.12, 0.15) 
    font_name = ImageFont.truetype(font_path, int(name_scale * height))
    
    draw.text((name_x, name_y), random.choice(names), font=font_name, fill=(0, 0, 0))

    # Chuyển ngược lại OpenCV format
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def add_random_vertical_lines(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Simulate the left/right border lines of a signature box cell.

    Real documents show 0, 1, or 2 vertical lines:
      - 0 lines : signature is NOT inside a box         (~30% of cases)
      - 1 line  : only one border visible in the crop   (~45% of cases)
      - 2 lines : full cell — both left and right edges (~25% of cases)

    The lines are placed near the edges (not centre) because Phase-1
    crops the signature tightly; a cell border appears close to the
    crop boundary, sometimes partially clipped.
    """
    num_lines = np.random.choice([0, 1, 2], p=[0.50, 0.35, 0.15])
    if num_lines == 0:
        return image

    # How far from each edge the border can land (0–15 % of width)
    edge_margin = int(0.15 * width)

    # Pick which borders appear
    if num_lines == 1:
        sides = [random.choice(["left", "right"])]
    else:
        sides = ["left", "right"]

    for side in sides:
        thickness = np.random.randint(1, 4)
        if side == "left":
            x = np.random.randint(0, max(1, edge_margin))
        else:
            x = np.random.randint(max(0, width - edge_margin), width)

        # The line rarely spans the full height — it can be clipped at top/bottom
        y_start = np.random.randint(0, int(0.10 * height))
        y_end   = np.random.randint(int(0.90 * height), height)

        image = cv2.line(image, (x, y_start), (x, y_end), (0, 0, 0), thickness=thickness)

    return image

# ─────────────────────────────────────────────────────────────────────────────
# Image resize / pad to square
# ─────────────────────────────────────────────────────────────────────────────

def make_square(img: Image.Image, target: int = 512) -> Image.Image:
    """
    Paste the signature centred in a white square of size target×target.
    Aspect ratio is preserved; no content is cropped.
    """
    img = img.convert("RGB")
    x, y = img.size
    size = max(target, x, y)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - x) // 2, (size - y) // 2))
    return canvas.resize((target, target), Image.LANCZOS)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    image_path: str,
    stamp_aug: StampAugmentor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (clean_array, noisy_array) both as BGR uint8.

    'clean' is the padded & resized original.
    'noisy' adds lines, text, and stamp on top of the clean version.
    """
    pil_img  = Image.open(image_path).convert("RGB")
    clean_sq = make_square(pil_img)
    clean_np = cv2.cvtColor(np.array(clean_sq), cv2.COLOR_RGB2BGR)

    h, w = clean_np.shape[:2]
    noisy = clean_np.copy()
    noisy = add_random_straight_lines(noisy, h, w)
    noisy = add_random_vertical_lines(noisy, h, w)
    noisy = add_random_text(noisy, h, w)
    noisy = stamp_aug(noisy)

    return clean_np, noisy


# ─────────────────────────────────────────────────────────────────────────────
# Dataset split & write
# ─────────────────────────────────────────────────────────────────────────────

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_images(src: str) -> List[str]:
    """Walk src recursively and return all image paths."""
    paths: List[str] = []
    for root, _, files in os.walk(src):
        for f in files:
            if Path(f).suffix.lower() in _VALID_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def make_dirs(dst: str) -> dict:
    dirs = {
        "trainA": os.path.join(dst, "trainA"),
        "trainB": os.path.join(dst, "trainB"),
        "testA":  os.path.join(dst, "testA"),
        "testB":  os.path.join(dst, "testB"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def build_dataset(
    src: str,
    dst: str,
    stamp_folder: str = "stamp_noise",
    test_ratio: float = 0.10,
    target_size: int = 512,
    seed: int = 42,
) -> None:
    """
    Full dataset builder.  Reads from `src`, writes CycleGAN folders to `dst`.
    """
    random.seed(seed)
    np.random.seed(seed)

    paths = collect_images(src)
    if not paths:
        raise FileNotFoundError(f"No images found under '{src}'")
    print(f"[dataset] Found {len(paths)} source images.")

    # Stamp augmentor (stamps loaded once, reused for all images)
    stamp_aug = StampAugmentor(stamp_folder=stamp_folder)
    if not stamp_aug._stamps:
        print("[dataset] WARNING: no stamp images found – stamp noise disabled.")

    dirs = make_dirs(dst)

    # Shuffle and split
    random.shuffle(paths)
    n_test  = max(1, int(len(paths) * test_ratio))
    test_paths  = paths[:n_test]
    train_paths = paths[n_test:]

    def _write(img_paths: List[str], split: str) -> None:
        dir_a = dirs[f"{split}A"]
        dir_b = dirs[f"{split}B"]
        for p in tqdm(img_paths, desc=f"  {split}"):
            stem = Path(p).stem
            try:
                clean, noisy = process_image(p, stamp_aug)
            except Exception as e:
                print(f"  [skip] {p}: {e}")
                continue
            cv2.imwrite(os.path.join(dir_a, f"{stem}.png"), clean)
            cv2.imwrite(os.path.join(dir_b, f"{stem}.png"), noisy)

    print("[dataset] Generating train pairs …")
    _write(train_paths, "train")
    print("[dataset] Generating test pairs …")
    _write(test_paths, "test")

    print(
        f"\n[dataset] Done.\n"
        f"  trainA/B: {len(train_paths)} pairs\n"
        f"  testA/B:  {len(test_paths)}  pairs\n"
        f"  Output  : {dst}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CycleGAN paired dataset.")
    p.add_argument("--src",     required=True,  help="Folder of clean signature images")
    p.add_argument("--dst",     required=True,  help="Output CycleGAN dataset folder")
    p.add_argument("--stamps",  default="stamp_noise", help="Folder with stamp images")
    p.add_argument("--split",   type=float, default=0.10, help="Test split ratio (default 0.10)")
    p.add_argument("--size",    type=int,   default=512,  help="Output image size (default 512)")
    p.add_argument("--seed",    type=int,   default=42,   help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        src=args.src,
        dst=args.dst,
        stamp_folder=args.stamps,
        test_ratio=args.split,
        target_size=args.size,
        seed=args.seed,
    )
