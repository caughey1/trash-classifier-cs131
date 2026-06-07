"""Print comparison table from saved run metrics."""

from __future__ import annotations

import argparse
import json

from config import RUNS_DIR


def load_accuracy(run_name: str) -> float | None:
    metrics_path = RUNS_DIR / run_name / "test_results" / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as handle:
        return float(json.load(handle)["accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare test accuracies across runs.")
    parser.add_argument("--baseline", default="resnet18_raw_baseline")
    parser.add_argument("--edges", default="resnet18_edges")
    parser.add_argument("--edge-overlay", default="resnet18_edge_overlay")
    parser.add_argument("--contour", default="resnet18_contour")
    parser.add_argument("--overlay", default="resnet18_contour_overlay")
    parser.add_argument("--fusion-edges", default="resnet18_fusion_edges")
    parser.add_argument(
        "--fusion-edge-overlay", default="resnet18_fusion_edge_overlay"
    )
    parser.add_argument("--fusion-contour", default="resnet18_fusion_contour")
    parser.add_argument(
        "--ensemble-baseline-fusion",
        default="ensemble_baseline_fusion_edges",
    )
    args = parser.parse_args()

    rows = [
        ("Raw baseline", args.baseline),
        ("Canny edges only", args.edges),
        ("Edge overlay", args.edge_overlay),
        ("Raw + edges fusion (6-ch)", args.fusion_edges),
        ("Raw + edge overlay fusion (6-ch)", args.fusion_edge_overlay),
        ("Contour only", args.contour),
        ("Contour overlay", args.overlay),
        ("Raw + contour fusion (6-ch)", args.fusion_contour),
        ("Ensemble (baseline + fusion edges)", args.ensemble_baseline_fusion),
    ]

    print("\n| Method | Test accuracy |")
    print("|--------|---------------|")
    for label, run_name in rows:
        acc = load_accuracy(run_name)
        acc_str = f"{acc * 100:.2f}%" if acc is not None else "not run yet"
        print(f"| {label} | {acc_str} |")


if __name__ == "__main__":
    main()
