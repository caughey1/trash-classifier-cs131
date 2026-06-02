"""
Batch edge detection preprocessor for TrashNet using Canny Edge Detector (Project 1).

Reads from:   data/dataset-resized/<class>/<file>.jpg
Writes to:    data/processed/<class>/<file>.jpg

Usage:
    python src/preprocess_edges.py
    python src/preprocess_edges.py --sigma 1.4 --high 20 --low 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import CLASS_NAMES, PROCESSED_DATA_DIR, RAW_DATA_DIR
from edge import canny


def process_image(img_path: Path, sigma: float, high: float, low: float) -> np.ndarray:
    """Load one image, run CS131 Canny, return a (H, W, 3) uint8 array."""
    
    # Load and convert to grayscale numpy array (what your canny() expects)
    img = Image.open(img_path).convert("L")  # "L" = grayscale
    img_array = np.array(img, dtype=np.float64)

    # Run your CS131 Canny — returns a boolean (H, W) array
    edges = canny(img_array, sigma=sigma, high=high, low=low)

    # Convert boolean → uint8 (True=255, False=0)
    edge_uint8 = (edges * 255).astype(np.uint8)

    # Stack to 3 channels so ResNet sees same format as raw images
    return np.stack([edge_uint8, edge_uint8, edge_uint8], axis=-1)


def process_class(
    class_name: str,
    raw_root: Path,
    out_root: Path,
    sigma: float,
    high: float,
    low: float,
) -> int:
    in_dir = raw_root / class_name
    out_dir = out_root / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.png"))
    if not images:
        print(f"  WARNING: no images found in {in_dir}")
        return 0

    for img_path in tqdm(images, desc=class_name, leave=False):
        out_path = out_dir / img_path.name
        if out_path.exists():
            continue  # safe to rerun
        edge_img = process_image(img_path, sigma, high, low)
        Image.fromarray(edge_img).save(out_path)

    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="CS131 Canny preprocessor for TrashNet.")
    parser.add_argument("--sigma", type=float, default=1.4, help="Gaussian sigma")
    parser.add_argument("--high",  type=float, default=20.0, help="High threshold")
    parser.add_argument("--low",   type=float, default=15.0, help="Low threshold")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data not found at {args.raw_dir}. Run src/download_data.py first."
        )

    print(f"Input:  {args.raw_dir}")
    print(f"Output: {args.out_dir}")
    print(f"Canny params: sigma={args.sigma}, high={args.high}, low={args.low}\n")

    total = 0
    for class_name in CLASS_NAMES:
        count = process_class(
            class_name, args.raw_dir, args.out_dir, args.sigma, args.high, args.low
        )
        print(f"  {class_name}: {count} images")
        total += count

    print(f"\nDone. {total} images written to {args.out_dir}")


if __name__ == "__main__":
    main()