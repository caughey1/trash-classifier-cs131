# Trash Classifier (CS131)

## project Overview

Waste classification is a common and costly problem. Most automated approaches treat this as a standard image classification task, feeding raw photos into a convolutional neural network. This project tests whether preprocessing with explicit edge information — extracted using a Canny edge detector — improves a neural network's ability to classify waste. 

Our base architecture was the ResNet18 in PyTorch. We trained on the Trashnet dataset: ~2500 images of waste, classified into 6 categories: glass, cardboard, paper, plastic, metal, trash.

We investigated 2 methods of pre-processing integration:
Base model — ResNet18 trained on raw RGB TrashNet images
Overlay model — Canny edge detector applied to grayscale images, edges overlaid on the original
Fusion model — raw RGB + Canny edge map stacked into a 6-channel input (3 RGB + 3 replicated edge channels), trained as a distinct model

Each model outputs a probability score across all 6 classes; final prediction is the class with the highest averaged score across all three models (ensemble averaging). The intuition: each model makes different mistakes depending on which visual cues it relies on (texture/color vs. shape/structure), so averaging cancels out some of those individual errors.

Our best result was outputted with a three-model ensemble: 95.8% accuracy, a 2.8-point improvement over baseline and a 20% reduction in total errors. The three-model creates a final prediction where each model outputs a probability score across all 6 classes; final prediction is the class with the highest averaged score across all three models (ensemble averaging).

## Setup (first time only)

You need Python 3.10+.

```bash
cd trash-classifier-cs131
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data and splits

```bash
python src/download_data.py
python src/split_data.py
```

Raw images land in `data/dataset-resized/`. Split files go in `splits/` (70/13/17).

## Preprocessing

**Canny edges (partner script)** — writes to `data/processed/` with class folders at the top level. Do not rename this folder.

```bash
python src/preprocess_edges.py
```

Output layout:

```
data/processed/
  glass/
  paper/
  cardboard/
  plastic/
  metal/
  trash/
```

**Edge overlay (partner Canny drawn on original photos):**

```bash
python src/preprocess_edge_overlay.py
```

**Contours (black bg, white outlines)** — separate subfolder:

```bash
python src/preprocess_contour.py
```

**Contour overlay (green lines on original photos):**

```bash
python src/preprocess_overlay.py
```

## Train the model

**Baseline (raw images):**

```bash
python src/train.py --input-type raw --epochs 15 --run-name resnet18_raw_baseline
```

**Canny edges only** (uses `data/processed/` — same paths as raw):

```bash
python src/train.py --input-type processed --epochs 15 --run-name resnet18_edges
```

**Contour only:**

```bash
python src/train.py --input-type processed --data-dir data/processed/contour --run-name resnet18_contour
```

**Edge overlay** (original photo + partner Canny edges in green):

```bash
python src/train.py --input-type raw --data-dir data/processed/edge_overlay --run-name resnet18_edge_overlay
```

**Contour overlay** (OpenCV contour lines — separate experiment):

```bash
python src/train.py --input-type raw --data-dir data/processed/contour_overlay --run-name resnet18_contour_overlay
```

**Fusion (raw + preprocessed, 6 channels):**

```bash
python src/train_fusion.py --processed edges --run-name resnet18_fusion_edges
python src/train_fusion.py --processed edges \
  --processed-dir data/processed/edge_overlay --run-name resnet18_fusion_edge_overlay
python src/train_fusion.py --processed contour --run-name resnet18_fusion_contour
```

Fusion training uses weight decay and early stopping by default (`--weight-decay 1e-4`,
`--patience 3`). Binary edge maps use `(x - 0.5) / 0.5` normalization; overlay images
use ImageNet stats (auto-detected from the processed path).

Compare saved runs:

```bash
python src/compare_results.py
```

**Ensemble** (average logits from multiple checkpoints; often beats single models):

```bash
python src/ensemble_eval.py \
  --checkpoints resnet18_raw_baseline_best.pt resnet18_fusion_edges_best.pt \
  --run-name ensemble_baseline_fusion_edges
```

## Check results on a saved model

```bash
python src/evaluate.py --checkpoint checkpoints/YOUR_RUN_NAME_best.pt --split test
```

## What's in the repo

- `src/preprocess_edges.py` / `src/preprocess_edge_overlay.py` — CS131 Canny (partner)
- `src/preprocess_contour.py` / `src/preprocess_overlay.py` — contour preprocessing
- `src/train.py` — ResNet18 on raw or processed-only
- `src/train_fusion.py` — ResNet18 on raw + processed fused
- `src/evaluate.py` — test a saved model

## citation

TrashNet: https://github.com/garythung/trashnet
