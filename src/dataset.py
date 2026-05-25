"""PyTorch Dataset for TrashNet (raw or preprocessed images)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, SPLITS_DIR


def resolve_image_path(data_root: Path, rel_path: str) -> Path:
    """Resolve a split entry like 'glass/glass189.jpg' to an on-disk path."""
    return data_root / rel_path


def load_split_file(split_path: Path) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    with split_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(" ", 1)
            entries.append((rel_path, int(label)))
    return entries


def build_transforms(split: str, input_type: str = "raw") -> transforms.Compose:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

    if input_type == "raw" and split == "train":
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ]
        )

    # Val/test and preprocessed inputs: no augmentation.
    if input_type != "raw":
        # Grayscale edge maps are expanded to 3 channels for ResNet.
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                resize,
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose([resize, transforms.ToTensor(), normalize])


class TrashNetDataset(Dataset):
    """Load TrashNet images listed in a split file."""

    def __init__(
        self,
        split: str,
        data_root: Path,
        split_dir: Path = SPLITS_DIR,
        input_type: str = "raw",
        transform: transforms.Compose | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.split = split
        self.data_root = data_root
        self.input_type = input_type
        self.entries = load_split_file(split_dir / f"{split}.txt")
        self.transform = transform or build_transforms(split, input_type)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rel_path, label = self.entries[index]
        image_path = resolve_image_path(self.data_root, rel_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def class_names(self) -> list[str]:
        return CLASS_NAMES
