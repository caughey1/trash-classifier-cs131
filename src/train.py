"""Train a ResNet18 baseline on TrashNet."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHECKPOINTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from dataset import TrashNetDataset
from evaluate import evaluate_model
from model_utils import build_model, get_device


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False, desc="train" if train else "eval"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def save_history(run_dir: Path, history: dict) -> None:
    with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 on TrashNet.")
    parser.add_argument(
        "--input-type",
        choices=("raw", "processed"),
        default="raw",
        help="Use raw TrashNet images or preprocessed images (same folder layout).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override dataset root (defaults based on --input-type).",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch instead of fine-tuning ImageNet weights.",
    )
    args = parser.parse_args()

    data_root = args.data_dir or (
        RAW_DATA_DIR if args.input_type == "raw" else PROCESSED_DATA_DIR
    )
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_root}. "
            "Run download/split scripts first, or point --data-dir to preprocessed images."
        )

    device = get_device()
    print(f"Using device: {device}")
    print(f"Data root: {data_root} ({args.input_type})")

    train_ds = TrashNetDataset("train", data_root, input_type=args.input_type)
    val_ds = TrashNetDataset("val", data_root, input_type=args.input_type)
    test_ds = TrashNetDataset("test", data_root, input_type=args.input_type)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"resnet18_{args.input_type}_{timestamp}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_checkpoint = CHECKPOINTS_DIR / f"{run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_type": args.input_type,
                    "class_names": train_ds.class_names,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_checkpoint,
            )

    save_history(run_dir, history)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to {best_checkpoint}")

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device, run_dir / "test_results")
    print(
        f"Test accuracy: {test_metrics['accuracy']:.4f} "
        f"(saved plots to {run_dir / 'test_results'})"
    )


if __name__ == "__main__":
    main()
