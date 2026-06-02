"""Export correct and incorrect test predictions for presentations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from config import CHECKPOINTS_DIR, CLASS_NAMES, IDX_TO_CLASS, RAW_DATA_DIR, RUNS_DIR
from dataset import TrashNetDataset, resolve_image_path
from model_utils import build_model, get_device


def predict_dataset(
    model: torch.nn.Module,
    dataset: TrashNetDataset,
    device: torch.device,
) -> list[dict]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    results: list[dict] = []
    offset = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu()
            confidences = probs.max(dim=1).values.cpu()

            for i in range(labels.size(0)):
                idx = offset + i
                rel_path, true_label = dataset.entries[idx]
                pred_label = int(preds[i])
                results.append(
                    {
                        "rel_path": rel_path,
                        "true_label": int(labels[i]),
                        "pred_label": pred_label,
                        "true_class": IDX_TO_CLASS[int(labels[i])],
                        "pred_class": IDX_TO_CLASS[pred_label],
                        "confidence": float(confidences[i]),
                        "correct": pred_label == int(labels[i]),
                    }
                )
            offset += labels.size(0)

    return results


def save_single_example(
    image_path: Path,
    result: dict,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.axis("off")

    status = "Correct" if result["correct"] else "Wrong"
    color = "#1a7f37" if result["correct"] else "#c41e3a"
    title = (
        f"{status}\n"
        f"True: {result['true_class']}\n"
        f"Predicted: {result['pred_class']} ({result['confidence']:.0%})"
    )
    ax.set_title(title, fontsize=14, fontweight="bold", color=color, pad=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_grid(
    examples: list[dict],
    data_root: Path,
    output_path: Path,
    title: str,
    cols: int = 3,
) -> None:
    if not examples:
        return

    n = len(examples)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, result in zip(axes_flat, examples):
        image_path = resolve_image_path(data_root, result["rel_path"])
        ax.imshow(Image.open(image_path).convert("RGB"))
        ax.axis("off")
        status = "OK" if result["correct"] else "FAIL"
        ax.set_title(
            f"{status}: {result['true_class']} -> {result['pred_class']}\n"
            f"({result['confidence']:.0%})",
            fontsize=10,
        )

    for ax in axes_flat[len(examples) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pass/fail examples for slides.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "resnet18_raw_baseline_best.pt",
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNS_DIR / "resnet18_raw_baseline" / "ppt_examples",
    )
    parser.add_argument(
        "--max-correct",
        type=int,
        default=12,
        help="Max correct examples to save individually (2 per class).",
    )
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(pretrained=not checkpoint.get("no_pretrained", False)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = TrashNetDataset("test", args.data_dir, input_type="raw")
    results = predict_dataset(model, dataset, device)

    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]

    out = args.output_dir
    correct_dir = out / "correct"
    incorrect_dir = out / "incorrect"

    # Save all misclassifications (usually ~20 at 95% acc)
    for i, result in enumerate(incorrect):
        path = resolve_image_path(args.data_dir, result["rel_path"])
        name = Path(result["rel_path"]).stem
        save_single_example(
            path,
            result,
            incorrect_dir / f"{i+1:02d}_{name}_true-{result['true_class']}_pred-{result['pred_class']}.png",
        )

    # Save diverse correct examples: up to 2 per class
    per_class: dict[str, list[dict]] = {name: [] for name in CLASS_NAMES}
    for result in sorted(correct, key=lambda r: -r["confidence"]):
        if len(per_class[result["true_class"]]) < 2:
            per_class[result["true_class"]].append(result)

    selected_correct = []
    for name in CLASS_NAMES:
        selected_correct.extend(per_class[name])
    selected_correct = selected_correct[: args.max_correct]

    for i, result in enumerate(selected_correct):
        path = resolve_image_path(args.data_dir, result["rel_path"])
        name = Path(result["rel_path"]).stem
        save_single_example(
            path,
            result,
            correct_dir / f"{i+1:02d}_{name}_{result['true_class']}.png",
        )

    # Summary grids for PPT
    save_grid(
        incorrect[:9],
        args.data_dir,
        out / "grid_failures.png",
        f"Misclassified examples ({len(incorrect)} total on test set)",
    )
    save_grid(
        selected_correct[:9],
        args.data_dir,
        out / "grid_correct.png",
        "Correctly classified examples",
    )

    summary = {
        "test_size": len(results),
        "num_correct": len(correct),
        "num_incorrect": len(incorrect),
        "accuracy": len(correct) / len(results),
        "incorrect_examples": incorrect,
        "correct_examples_saved": [
            {k: v for k, v in r.items() if k != "rel_path"} | {"file": r["rel_path"]}
            for r in selected_correct
        ],
    }
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Test set: {len(results)} images")
    print(f"Correct: {len(correct)} | Incorrect: {len(incorrect)}")
    print(f"Saved to {out}")
    print(f"  Individual failures: {incorrect_dir}/ ({len(incorrect)} images)")
    print(f"  Individual correct:  {correct_dir}/ ({len(selected_correct)} images)")
    print(f"  Grids: grid_failures.png, grid_correct.png")


if __name__ == "__main__":
    main()
