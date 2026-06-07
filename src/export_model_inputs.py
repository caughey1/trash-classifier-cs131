"""Export side-by-side images showing what each model type sees (for slides)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import (
    CLASS_NAMES,
    CONTOUR_OVERLAY_DATA_DIR,
    EDGE_OVERLAY_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)

# Hand-picked readable examples (one per class where possible)
EXAMPLES = [
    ("glass/glass1.jpg", "glass"),
    ("paper/paper10.jpg", "paper"),
    ("cardboard/cardboard1.jpg", "cardboard"),
    ("plastic/plastic5.jpg", "plastic"),
    ("metal/metal5.jpg", "metal"),
    ("trash/trash1.jpg", "trash"),
]


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def make_labeled_panel(
    images: list[tuple[str, np.ndarray]],
    suptitle: str,
    output_path: Path,
    ncols: int | None = None,
) -> None:
    n = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (label, img) in zip(axes_flat, images):
        ax.imshow(img)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    for ax in axes_flat[len(images) :]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fusion_diagram(output_path: Path, example_rel: str = "metal/metal5.jpg") -> None:
    raw = load_rgb(RAW_DATA_DIR / example_rel)
    edges = load_rgb(PROCESSED_DATA_DIR / example_rel)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))

    axes[0].imshow(raw)
    axes[0].set_title("Channels 1–3\nRaw RGB photo", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Channels 4–6\nCanny edge map\n(×3 as grayscale RGB)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].axis("off")
    axes[2].text(
        0.5,
        0.55,
        "stack\n→",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        transform=axes[2].transAxes,
    )
    axes[2].text(
        0.5,
        0.25,
        "6-channel input\n→ Fusion ResNet18",
        ha="center",
        va="center",
        fontsize=12,
        transform=axes[2].transAxes,
    )

    # Visual stack: raw | edges side-by-side with bracket
    combo = np.concatenate([raw, edges], axis=1)
    axes[3].imshow(combo)
    axes[3].set_title("What fusion sees (conceptually)", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    w = raw.shape[1]
    axes[3].axvline(w - 0.5, color="white", linewidth=2, linestyle="--")

    fig.suptitle("Fusion model input (6 channels)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_ensemble_diagram(output_path: Path, example_rel: str = "metal/metal5.jpg") -> None:
    raw = load_rgb(RAW_DATA_DIR / example_rel)
    edges = load_rgb(PROCESSED_DATA_DIR / example_rel)
    overlay = load_rgb(EDGE_OVERLAY_DATA_DIR / example_rel)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    panels = [
        ("Baseline model\n(raw photo)", raw),
        ("Fusion model\n(6-ch: photo + edges)", np.concatenate([raw, edges], axis=1)),
        ("Edge overlay model\n(photo + Canny edges)", overlay),
    ]
    for ax, (title, img) in zip(axes[:3], panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")

    axes[3].axis("off")
    axes[3].text(
        0.5,
        0.6,
        "Average class scores",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        transform=axes[3].transAxes,
    )
    axes[3].text(
        0.5,
        0.35,
        "→ pick highest\n(ensemble @ inference)",
        ha="center",
        va="center",
        fontsize=12,
        transform=axes[3].transAxes,
    )

    fig.suptitle(
        "Ensemble (95.8%): three separate models, same photo",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model-input figures for PPT.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "ppt_model_inputs",
    )
    args = parser.parse_args()
    out = args.output_dir

    # One row: all input types for a single example
    example = "metal/metal5.jpg"
    make_labeled_panel(
        [
            ("Baseline\n(raw photo)", load_rgb(RAW_DATA_DIR / example)),
            ("Edges only\n(Canny map)", load_rgb(PROCESSED_DATA_DIR / example)),
            ("Edge overlay\n(Canny on photo)", load_rgb(EDGE_OVERLAY_DATA_DIR / example)),
            (
                "Contour overlay\n(OpenCV on photo)",
                load_rgb(CONTOUR_OVERLAY_DATA_DIR / example),
            ),
        ],
        f"Four input types — example: {example}",
        out / "01_all_input_types_one_example.png",
    )

    # Grid: raw vs edge overlay across classes
    class_panels = []
    for rel, cls in EXAMPLES:
        class_panels.append((f"Raw — {cls}", load_rgb(RAW_DATA_DIR / rel)))
    make_labeled_panel(
        class_panels,
        "Baseline model sees: raw TrashNet photos",
        out / "02_baseline_raw_by_class.png",
        ncols=3,
    )

    overlay_panels = []
    for rel, cls in EXAMPLES:
        overlay_panels.append(
            (f"Edge overlay — {cls}", load_rgb(EDGE_OVERLAY_DATA_DIR / rel))
        )
    make_labeled_panel(
        overlay_panels,
        "Overlay model sees: photo + green Canny edges",
        out / "03_edge_overlay_by_class.png",
        ncols=3,
    )

    edge_panels = []
    for rel, cls in EXAMPLES:
        edge_panels.append((f"Canny edges — {cls}", load_rgb(PROCESSED_DATA_DIR / rel)))
    make_labeled_panel(
        edge_panels,
        "Edges-only / fusion edge branch: partner Canny maps",
        out / "04_canny_edges_by_class.png",
        ncols=3,
    )

    make_fusion_diagram(out / "05_fusion_6channel_diagram.png", example)
    make_ensemble_diagram(out / "06_ensemble_three_models.png", example)

    # Individual files for flexible PPT layout
    indiv = out / "individual"
    indiv.mkdir(parents=True, exist_ok=True)
    for rel, cls in EXAMPLES:
        stem = Path(rel).stem
        for name, root in [
            ("raw", RAW_DATA_DIR),
            ("edges", PROCESSED_DATA_DIR),
            ("edge_overlay", EDGE_OVERLAY_DATA_DIR),
            ("contour_overlay", CONTOUR_OVERLAY_DATA_DIR),
        ]:
            src = root / rel
            if src.exists():
                Image.open(src).convert("RGB").save(indiv / f"{cls}_{stem}_{name}.jpg")

    print(f"Saved PPT figures to {out}/")
    print("  01_all_input_types_one_example.png")
    print("  02_baseline_raw_by_class.png")
    print("  03_edge_overlay_by_class.png")
    print("  04_canny_edges_by_class.png")
    print("  05_fusion_6channel_diagram.png")
    print("  06_ensemble_three_models.png")
    print(f"  individual/  ({len(list(indiv.glob('*.jpg')))} files)")


if __name__ == "__main__":
    main()
