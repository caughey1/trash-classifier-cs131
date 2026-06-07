"""Train ResNet18 on fused raw + preprocessed images (6 channels)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR, CONTOUR_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from evaluate import evaluate_model
from fusion_dataset import TrashNetFusionDataset, infer_processed_normalize
from model_utils import build_fusion_model, get_device
from train import run_epoch, save_history


def resolve_processed_dir(kind: str, custom: Path | None) -> Path:
    if custom is not None:
        return custom
    if kind == "edges":
        return PROCESSED_DATA_DIR
    if kind == "contour":
        return CONTOUR_DATA_DIR
    raise ValueError(f"Unknown processed kind: {kind!r}. Use edges or contour.")


def preprocess_hint(kind: str) -> str:
    if kind == "edges":
        return "python src/preprocess_edges.py"
    return "python src/preprocess_contour.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ResNet18 on raw+processed fusion (6 channels)."
    )
    parser.add_argument(
        "--processed",
        choices=("edges", "contour"),
        required=True,
        help="edges = data/processed/ (Canny), contour = data/processed/contour/",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Override processed root.",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train fusion model without ImageNet weights.",
    )
    parser.add_argument(
        "--processed-normalize",
        choices=("imagenet", "binary"),
        default=None,
        help="Normalization for processed branch (default: imagenet for overlay, binary for edges).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam L2 weight decay.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop if val accuracy does not improve for this many epochs (0 = disabled).",
    )
    args = parser.parse_args()

    processed_root = resolve_processed_dir(args.processed, args.processed_dir)
    processed_normalize = args.processed_normalize or infer_processed_normalize(
        processed_root
    )
    if not processed_root.exists():
        raise FileNotFoundError(
            f"Processed images not found at {processed_root}.\n"
            f"Run: {preprocess_hint(args.processed)}"
        )
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset not found at {args.raw_dir}")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Fusion: raw={args.raw_dir} + processed={processed_root}")
    print(f"Processed normalize: {processed_normalize}")

    ds_kwargs = {
        "raw_root": args.raw_dir,
        "processed_root": processed_root,
        "processed_normalize": processed_normalize,
    }
    train_ds = TrashNetFusionDataset("train", **ds_kwargs)
    val_ds = TrashNetFusionDataset("val", **ds_kwargs)
    test_ds = TrashNetFusionDataset("test", **ds_kwargs)

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

    model = build_fusion_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"resnet18_fusion_{args.processed}_{timestamp}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
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
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fusion": True,
                    "processed": args.processed,
                    "processed_dir": str(processed_root),
                    "processed_normalize": processed_normalize,
                    "class_names": train_ds.class_names,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val improvement for {args.patience} epochs)"
                )
                break

    save_history(run_dir, history)
    print(f"\nBest val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Checkpoint saved to {best_checkpoint}")

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device, run_dir / "test_results")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
