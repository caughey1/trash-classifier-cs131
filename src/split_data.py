"""Convert official TrashNet splits to project split files (0-based labels)."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    LUA_LABEL_TO_CLASS,
    RAW_DATA_DIR,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from dataset import resolve_image_path


def parse_official_line(line: str) -> tuple[str, int]:
    """Parse 'glass189.jpg 1' into (relative_path, 0-based label)."""
    parts = line.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid split line: {line!r}")

    filename, lua_label = parts[0], int(parts[1])
    class_name = LUA_LABEL_TO_CLASS[lua_label]
    rel_path = f"{class_name}/{filename}"
    label = CLASS_TO_IDX[class_name]
    return rel_path, label


def write_split(path: Path, entries: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rel_path, label in entries:
            handle.write(f"{rel_path} {label}\n")


def load_official_splits() -> dict[str, list[tuple[str, int]]]:
    splits: dict[str, list[tuple[str, int]]] = {}
    for split_name in ("train", "val", "test"):
        official_path = SPLITS_DIR / f"official_{split_name}.txt"
        if not official_path.exists():
            raise FileNotFoundError(
                f"Missing {official_path}. Run: python src/download_data.py"
            )
        entries = [parse_official_line(line) for line in official_path.read_text().splitlines() if line.strip()]
        splits[split_name] = entries
    return splits


def create_stratified_splits(
    data_dir: Path,
    seed: int,
) -> dict[str, list[tuple[str, int]]]:
    """Create a fresh 70/13/17 stratified split from all images on disk."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                by_class[class_name].append(f"{class_name}/{image_path.name}")

    rng = random.Random(seed)
    train_entries: list[tuple[str, int]] = []
    val_entries: list[tuple[str, int]] = []
    test_entries: list[tuple[str, int]] = []

    for class_name in CLASS_NAMES:
        paths = by_class[class_name]
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        for rel_path in paths[:n_train]:
            train_entries.append((rel_path, CLASS_TO_IDX[class_name]))
        for rel_path in paths[n_train : n_train + n_val]:
            val_entries.append((rel_path, CLASS_TO_IDX[class_name]))
        for rel_path in paths[n_train + n_val :]:
            test_entries.append((rel_path, CLASS_TO_IDX[class_name]))

        assert n_test >= 0

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    rng.shuffle(test_entries)

    return {"train": train_entries, "val": val_entries, "test": test_entries}


def validate_entries(
    entries: list[tuple[str, int]],
    data_dir: Path,
    split_name: str,
) -> None:
    missing = []
    for rel_path, label in entries:
        if not resolve_image_path(data_dir, rel_path).exists():
            missing.append(rel_path)
    if missing:
        sample = missing[:5]
        raise FileNotFoundError(
            f"{split_name} split references {len(missing)} missing files. "
            f"Examples: {sample}"
        )


def print_split_summary(splits: dict[str, list[tuple[str, int]]]) -> None:
    print("\nSplit summary:")
    for split_name, entries in splits.items():
        counts = Counter(CLASS_NAMES[label] for _, label in entries)
        print(f"  {split_name:5s}: {len(entries):4d} images  {dict(counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create TrashNet train/val/test split files.")
    parser.add_argument(
        "--mode",
        choices=("official", "stratified"),
        default="official",
        help="Use TrashNet's published split (default) or create a new stratified split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified mode.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root directory containing class subfolders.",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.data_dir}. Run: python src/download_data.py"
        )

    if args.mode == "official":
        splits = load_official_splits()
    else:
        splits = create_stratified_splits(args.data_dir, args.seed)

    for split_name, entries in splits.items():
        validate_entries(entries, args.data_dir, split_name)
        write_split(SPLITS_DIR / f"{split_name}.txt", entries)

    print_split_summary(splits)
    print(f"\nWrote split files to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
