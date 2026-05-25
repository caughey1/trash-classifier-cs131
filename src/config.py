"""Shared paths and TrashNet class definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "dataset-resized"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROJECT_ROOT / "splits"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"

# 0-based label order matches TrashNet constants.py (GLASS=0 ... TRASH=5)
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

# Original TrashNet Lua loader uses 1-based labels; we use 0-based in PyTorch.
LUA_LABEL_TO_CLASS = {
    1: "glass",
    2: "paper",
    3: "cardboard",
    4: "plastic",
    5: "metal",
    6: "trash",
}

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# 70 / 13 / 17 split (matches original TrashNet paper)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.13
TEST_RATIO = 0.17

TRASHNET_REPO = "https://github.com/garythung/trashnet"
DATASET_ZIP_URL = (
    "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
)
OFFICIAL_SPLITS = {
    "train": "https://raw.githubusercontent.com/garythung/trashnet/master/data/one-indexed-files-notrash_train.txt",
    "val": "https://raw.githubusercontent.com/garythung/trashnet/master/data/one-indexed-files-notrash_val.txt",
    "test": "https://raw.githubusercontent.com/garythung/trashnet/master/data/one-indexed-files-notrash_test.txt",
}

# ResNet expects 224x224 with ImageNet normalization
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
