"""Draw contour lines on top of the original TrashNet images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from config import CONTOUR_OVERLAY_DATA_DIR, RAW_DATA_DIR
from contour_utils import detect_contours, load_bgr
from preprocess_utils import list_dataset_rel_paths


def contour_overlay_rgb(
    image_path: Path,
    blur_ksize: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    line_thickness: int = 2,
    line_color_bgr: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    bgr = load_bgr(image_path)
    overlay = bgr.copy()
    contours = detect_contours(bgr, blur_ksize, canny_low, canny_high)
    if contours:
        cv2.drawContours(overlay, contours, -1, line_color_bgr, line_thickness)
    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw contour lines on original TrashNet images."
    )
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=CONTOUR_OVERLAY_DATA_DIR)
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

    for rel_path in tqdm(rel_paths, desc="Contour overlay"):
        src = args.input_dir / rel_path
        dst = args.output_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        out = contour_overlay_rgb(
            src,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            line_thickness=args.line_thickness,
        )
        out.save(dst)

    print(f"Saved {len(rel_paths)} overlay images to {args.output_dir}")


if __name__ == "__main__":
    main()
