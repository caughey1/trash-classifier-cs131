"""Shared helpers for batch image preprocessing."""

from __future__ import annotations

from config import SPLITS_DIR
from dataset import load_split_file


def list_dataset_rel_paths() -> list[str]:
    """All image paths listed in train/val/test split files."""
    paths: list[str] = []
    for split_file in ("train.txt", "val.txt", "test.txt"):
        for rel_path, _ in load_split_file(SPLITS_DIR / split_file):
            paths.append(rel_path)
    return sorted(set(paths))
