"""Draw object contours on TrashNet images (black background, white outlines)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import CONTOUR_DATA_DIR, RAW_DATA_DIR
from contour_utils import detect_contours, load_bgr
from preprocess_utils import list_dataset_rel_paths


def contour_rgb(
    image_path: Path,
    blur_ksize: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    line_thickness: int = 2,
) -> Image.Image:
    bgr = load_bgr(image_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros_like(gray)
    contours = detect_contours(bgr, blur_ksize, canny_low, canny_high)
    if contours:
        cv2.drawContours(canvas, contours, -1, 255, line_thickness)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contour images for TrashNet.")
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=CONTOUR_DATA_DIR)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--line-thickness", type=int, default=2)
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {args.input_dir}. Run: python src/download_data.py"
        )

    rel_paths = list_dataset_rel_paths()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for rel_path in tqdm(rel_paths, desc="Contour"):
        src = args.input_dir / rel_path
        dst = args.output_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        out = contour_rgb(
            src,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            line_thickness=args.line_thickness,
        )
        out.save(dst)

    print(f"Saved {len(rel_paths)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
