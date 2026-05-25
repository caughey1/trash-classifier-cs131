"""Evaluate a trained TrashNet classifier."""

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

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from dataset import TrashNetDataset
from model_utils import build_model, get_device


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, float | list[list[int]] | str]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    class_names = loader.dataset.class_names  # type: ignore[attr-defined]
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
    )
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))

    metrics: dict[str, float | list[list[int]] | str] = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "accuracy": accuracy,
                    "classification_report": report,
                    "confusion_matrix": cm.tolist(),
                    "class_names": class_names,
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
            title=f"Confusion Matrix (acc={accuracy:.3f})",
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

    print(report)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ResNet18 checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a .pt checkpoint saved by train.py",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument(
        "--input-type",
        choices=("raw", "processed"),
        default=None,
        help="Defaults to the value stored in the checkpoint.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Must match how the checkpoint was trained.",
    )
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    input_type = args.input_type or checkpoint.get("input_type", "raw")
    data_root = args.data_dir or (
        RAW_DATA_DIR if input_type == "raw" else PROCESSED_DATA_DIR
    )

    dataset = TrashNetDataset(args.split, data_root, input_type=input_type)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    run_name = args.checkpoint.stem.replace("_best", "")
    output_dir = args.output_dir or (RUNS_DIR / run_name / f"{args.split}_results")
    evaluate_model(model, loader, device, output_dir)


if __name__ == "__main__":
    main()
