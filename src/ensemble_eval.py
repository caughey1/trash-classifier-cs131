"""Ensemble test-set predictions from multiple checkpoints (average logits)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config import (
    CHECKPOINTS_DIR,
    CONTOUR_OVERLAY_DATA_DIR,
    EDGE_OVERLAY_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RUNS_DIR,
)
from dataset import TrashNetDataset
from fusion_dataset import TrashNetFusionDataset, infer_processed_normalize
from model_utils import build_fusion_model, build_model, get_device


def _data_root_for_checkpoint(checkpoint_path: Path, checkpoint: dict) -> Path:
    if checkpoint.get("fusion"):
        return Path(checkpoint["processed_dir"])
    input_type = checkpoint.get("input_type", "raw")
    if input_type != "raw":
        return PROCESSED_DATA_DIR
    stem = checkpoint_path.stem
    if "edge_overlay" in stem and "fusion" not in stem:
        return EDGE_OVERLAY_DATA_DIR
    if "contour_overlay" in stem:
        return CONTOUR_OVERLAY_DATA_DIR
    if "contour" in stem:
        return Path("data/processed/contour")
    return RAW_DATA_DIR


def build_loader_for_checkpoint(
    checkpoint_path: Path, split: str, batch_size: int
) -> DataLoader:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("fusion"):
        processed_dir = Path(checkpoint["processed_dir"])
        processed_normalize = checkpoint.get(
            "processed_normalize", infer_processed_normalize(processed_dir)
        )
        dataset = TrashNetFusionDataset(
            split,
            RAW_DATA_DIR,
            processed_dir,
            processed_normalize=processed_normalize,
        )
    else:
        input_type = checkpoint.get("input_type", "raw")
        data_root = _data_root_for_checkpoint(checkpoint_path, checkpoint)
        dataset = TrashNetDataset(split, data_root, input_type=input_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("fusion"):
        model = build_fusion_model(pretrained=False).to(device)
    else:
        model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble checkpoints on test set.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to .pt checkpoints to average.",
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-name", type=str, default="ensemble")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    device = get_device()
    paths = [p if p.is_absolute() else CHECKPOINTS_DIR / p for p in args.checkpoints]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    models = [load_model_from_checkpoint(p, device) for p in paths]
    loaders = [
        build_loader_for_checkpoint(p, args.split, args.batch_size) for p in paths
    ]
    class_names = loaders[0].dataset.class_names  # type: ignore[attr-defined]

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batches in zip(*[iter(loader) for loader in loaders]):
            images_list = [b[0].to(device) for b in batches]
            labels = batches[0][1]
            logits_sum = torch.zeros(
                images_list[0].size(0), len(class_names), device=device
            )
            for model, images in zip(models, images_list):
                logits_sum += model(images)
            logits_sum /= len(models)
            preds = logits_sum.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Ensemble ({len(paths)} models) {args.split} accuracy: {accuracy:.4f}")

    output_dir = args.output_dir or (
        RUNS_DIR / args.run_name / f"{args.split}_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "class_names": class_names,
                "checkpoints": [str(p) for p in paths],
            },
            handle,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Ensemble Confusion Matrix (acc={accuracy:.3f})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
