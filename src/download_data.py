"""Download TrashNet resized images and official train/val/test split files."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    DATA_DIR,
    DATASET_ZIP_URL,
    OFFICIAL_SPLITS,
    RAW_DATA_DIR,
    SPLITS_DIR,
    TRASHNET_REPO,
)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already exists, skipping: {dest}")
        return

    print(f"Downloading {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with dest.open("wb") as handle, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                progress.update(len(chunk))


def download_official_splits() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, url in OFFICIAL_SPLITS.items():
        dest = SPLITS_DIR / f"official_{split_name}.txt"
        download_file(url, dest)


def verify_dataset(data_dir: Path) -> None:
    expected_classes = {"glass", "paper", "cardboard", "plastic", "metal", "trash"}
    found_classes = {path.name for path in data_dir.iterdir() if path.is_dir()}
    missing = expected_classes - found_classes
    if missing:
        raise FileNotFoundError(
            f"Missing class folders in {data_dir}: {sorted(missing)}"
        )

    image_count = sum(
        1
        for class_dir in data_dir.iterdir()
        if class_dir.is_dir()
        for path in class_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Found {image_count} images across {len(found_classes)} classes in {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TrashNet dataset and official split files."
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Only download official split files (images already present).",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_images:
        zip_path = DATA_DIR / "dataset-resized.zip"
        download_file(DATASET_ZIP_URL, zip_path)

        if not RAW_DATA_DIR.exists():
            print(f"Extracting {zip_path.name} -> {DATA_DIR}")
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(DATA_DIR)
        else:
            print(f"Dataset already present at {RAW_DATA_DIR}")

        if not RAW_DATA_DIR.exists():
            candidates = list(DATA_DIR.glob("**/dataset-resized"))
            if len(candidates) == 1 and candidates[0] != RAW_DATA_DIR:
                candidates[0].rename(RAW_DATA_DIR)
            elif not RAW_DATA_DIR.exists():
                raise FileNotFoundError(
                    f"Could not find extracted dataset at {RAW_DATA_DIR}. "
                    f"See {TRASHNET_REPO} for manual download instructions."
                )

        verify_dataset(RAW_DATA_DIR)

    download_official_splits()
    print("\nDone.")
    print(f"  Raw images: {RAW_DATA_DIR}")
    print(f"  Official splits: {SPLITS_DIR}")
    print("\nNext step:")
    print("  python src/split_data.py")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
