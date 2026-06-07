"""Dataset that stacks raw RGB + preprocessed RGB into 6-channel tensors."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, RAW_DATA_DIR, SPLITS_DIR
from dataset import load_split_file, resolve_image_path


def _normalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def _normalize_binary(tensor: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] edge maps to [-1, 1]; ImageNet stats are a poor fit for binary edges."""
    return (tensor - 0.5) / 0.5


def infer_processed_normalize(processed_root: Path) -> str:
    """Overlay images are natural RGB; pure edge/contour maps are binary."""
    name = processed_root.name.lower()
    if "overlay" in name:
        return "imagenet"
    return "binary"


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(image.convert("RGB"))


class TrashNetFusionDataset(Dataset):
    """
    Each sample is 6 channels: [raw R,G,B | processed R,G,B].
    Raw and processed images must share the same relative path under their roots.
    """

    def __init__(
        self,
        split: str,
        raw_root: Path = RAW_DATA_DIR,
        processed_root: Path | None = None,
        split_dir: Path = SPLITS_DIR,
        processed_normalize: str | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if processed_root is None:
            raise ValueError("processed_root is required (e.g. data/processed)")

        norm = processed_normalize or infer_processed_normalize(processed_root)
        if norm not in {"imagenet", "binary"}:
            raise ValueError(f"processed_normalize must be imagenet or binary, got {norm!r}")

        self.split = split
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.processed_normalize = norm
        self.entries = load_split_file(split_dir / f"{split}.txt")

    def __len__(self) -> int:
        return len(self.entries)

    def _load_pair(self, rel_path: str) -> tuple[Image.Image, Image.Image]:
        raw_path = resolve_image_path(self.raw_root, rel_path)
        proc_path = resolve_image_path(self.processed_root, rel_path)
        if not proc_path.exists():
            raise FileNotFoundError(
                f"Missing processed image: {proc_path}\n"
                "Run preprocessing first (preprocess_edges.py or preprocess_contour.py)."
            )
        return Image.open(raw_path).convert("RGB"), Image.open(proc_path).convert("RGB")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rel_path, label = self.entries[index]
        raw_img, proc_img = self._load_pair(rel_path)

        raw_img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(raw_img)
        proc_img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(proc_img)

        if self.split == "train":
            if random.random() < 0.5:
                raw_img = F.hflip(raw_img)
                proc_img = F.hflip(proc_img)
            angle = random.uniform(-15, 15)
            raw_img = F.rotate(raw_img, angle)
            proc_img = F.rotate(proc_img, angle)

        raw_t = _normalize_imagenet(_pil_to_tensor(raw_img))
        proc_t = _pil_to_tensor(proc_img)
        if self.processed_normalize == "binary":
            proc_t = _normalize_binary(proc_t)
        else:
            proc_t = _normalize_imagenet(proc_t)
        fused = torch.cat([raw_t, proc_t], dim=0)
        return fused, label

    @property
    def class_names(self) -> list[str]:
        return CLASS_NAMES
