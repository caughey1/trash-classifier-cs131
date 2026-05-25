# Trash Classifier (CS131)

## What this project is

Most people who use the [TrashNet](https://github.com/garythung/trashnet) dataset feed the photos straight into a neural network and call it a day. We're expanding on this by running CS131 computer vision steps on the images first (edge detection, contouring, and so on), then train the same kind of classifier and see if that helps.

So we have two paths:

1. **Baseline** — normal TrashNet photos → ResNet → predict the trash type  
2. **Our approach** — preprocessed photos (e.g. Canny edges) → same ResNet → predict the trash type  

If preprocessing helps, we can say the extra CV layer was worth it. If not, that's still a useful result.

TrashNet has about 2,500 images in six categories: glass, paper, cardboard, plastic, metal, and trash.

## Setup (first time only)

You need Python 3.10+.

```bash
cd trash-classifier-cs131
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data and splits

**Step 1 — download the images** (~40 MB zip):

```bash
python src/download_data.py
```

This puts images in `data/dataset-resized/` (one folder per class).

**Step 2 — build train / val / test lists:**

```bash
python src/split_data.py
```

That writes three files in `splits/`: about 70% train, 13% validation, 17% test. We use TrashNet's official split so our numbers are comparable to other work.

## Train the model

**Baseline (raw images):**

```bash
python src/train.py --input-type raw --epochs 15
```

**After preprocessing is ready** (edge maps, etc. in `data/processed/` with the same folder names and filenames as raw):

```bash
python src/train.py --input-type processed --epochs 15
```

Training saves the best model in `checkpoints/` and results (accuracy, confusion matrix) in `runs/`.

On a Mac, if you get an SSL error when downloading ImageNet weights, try:

```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

Then run training again.

## Check results on a saved model

```bash
python src/evaluate.py --checkpoint checkpoints/YOUR_RUN_NAME_best.pt --split test
```

## Where preprocessed images go

Whoever runs the edge detector (or other preprocessing) should save output like this:

```
data/processed/
  glass/
  paper/
  cardboard/
  plastic/
  metal/
  trash/
```

Filenames should match the raw dataset (e.g. `glass/glass189.jpg`). The training code uses the same train/val/test lists for both raw and processed — you only change `--input-type`.

## What's in the repo

- `src/download_data.py` — download TrashNet  
- `src/split_data.py` — create split files  
- `src/train.py` — train ResNet18  
- `src/evaluate.py` — test a saved model  
- `data/` — images (not committed to git; run download script locally)  
- `splits/` — which images are in train / val / test  
- `checkpoints/` — trained models  
- `runs/` — metrics and plots  

## citation

TrashNet: https://github.com/garythung/trashnet 
