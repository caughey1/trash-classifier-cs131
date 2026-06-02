"""Quick visual check of Canny edge detection across all classes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from edge import canny
from config import CLASS_NAMES, RAW_DATA_DIR


def main():
    n_samples = 5
    n_classes = len(CLASS_NAMES)

    fig, axes = plt.subplots(
        n_classes, n_samples * 2,
        figsize=(n_samples * 4, n_classes * 2.5)
    )

    for row, class_name in enumerate(CLASS_NAMES):
        class_dir = RAW_DATA_DIR / class_name
        images = sorted(class_dir.glob("*.jpg"))[:n_samples]

        for col, img_path in enumerate(images):
            img = Image.open(img_path).convert("L")
            arr = np.array(img, dtype=np.float64)
            edges = canny(arr)

            # Original
            ax_orig = axes[row, col * 2]
            ax_orig.imshow(arr, cmap="gray")
            ax_orig.axis("off")
            if col == 0:
                ax_orig.set_ylabel(class_name, fontsize=11, rotation=90, labelpad=10)

            # Edge
            ax_edge = axes[row, col * 2 + 1]
            ax_edge.imshow(edges, cmap="gray")
            ax_edge.axis("off")

        # Column headers on first row
        if row == 0:
            for col in range(n_samples):
                axes[0, col * 2].set_title(f"orig {col+1}", fontsize=8)
                axes[0, col * 2 + 1].set_title(f"edge {col+1}", fontsize=8)

    plt.suptitle("Canny Edge Detection — 5 samples per class", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = Path("edge_preview.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()