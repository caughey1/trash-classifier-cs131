"""
Draw partner Canny edges on top of original TrashNet photos.

Reads from:   data/dataset-resized/<class>/<file>.jpg
Writes to:    data/processed/edge_overlay/<class>/<file>.jpg

Uses the same edge.canny() implementation as preprocess_edges.py.

Usage:
    python src/preprocess_edge_overlay.py
    python src/preprocess_edge_overlay.py --sigma 1.4 --high 5 --low 0
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import CLASS_NAMES, EDGE_OVERLAY_DATA_DIR, RAW_DATA_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge import canny


def process_image(args: tuple) -> str | None:
    img_path, out_path, sigma, high, low, edge_color = args

    if out_path.exists():
        return None

    try:
        rgb = np.array(Image.open(img_path).convert("RGB"))
        gray = np.array(Image.open(img_path).convert("L"), dtype=np.float64)
        edges = canny(gray, sigma=sigma, high=high, low=low)

        overlay = rgb.copy()
        mask = edges.astype(bool)
        overlay[mask] = edge_color
        Image.fromarray(overlay).save(out_path)
        return str(out_path)
    except Exception as e:
        print(f"  ERROR on {img_path.name}: {e}")
        return None


def process_class(
    class_name: str,
    raw_root: Path,
    out_root: Path,
    sigma: float,
    high: float,
    low: float,
    edge_color: tuple[int, int, int],
    workers: int,
) -> int:
    in_dir = raw_root / class_name
    out_dir = out_root / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.png"))
    if not images:
        print(f"  WARNING: no images found in {in_dir}")
        return 0

    job_args = [
        (img_path, out_dir / img_path.name, sigma, high, low, edge_color)
        for img_path in images
    ]

    with Pool(processes=workers) as pool:
        list(tqdm(pool.imap(process_image, job_args), total=len(images), desc=class_name))

    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay CS131 Canny edges on original TrashNet photos."
    )
    parser.add_argument("--sigma", type=float, default=1.4)
    parser.add_argument("--high", type=float, default=5.0)
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=EDGE_OVERLAY_DATA_DIR)
    parser.add_argument(
        "--edge-color",
        type=int,
        nargs=3,
        default=(0, 255, 0),
        metavar=("R", "G", "B"),
        help="RGB color for edge pixels (default: green)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help=f"Parallel workers (default: all {cpu_count()} cores)",
    )
    args = parser.parse_args()
    edge_color = tuple(args.edge_color)

    if not args.raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data not found at {args.raw_dir}. Run src/download_data.py first."
        )

    print(f"Input:   {args.raw_dir}")
    print(f"Output:  {args.out_dir}")
    print(f"Canny:   sigma={args.sigma}, high={args.high}, low={args.low}")
    print(f"Color:   RGB{edge_color}")
    print(f"Workers: {args.workers} cores\n")

    total = 0
    for class_name in CLASS_NAMES:
        count = process_class(
            class_name,
            args.raw_dir,
            args.out_dir,
            args.sigma,
            args.high,
            args.low,
            edge_color,
            args.workers,
        )
        print(f"  {class_name}: {count} images")
        total += count

    print(f"\nDone. {total} overlay images written to {args.out_dir}")


if __name__ == "__main__":
    main()
