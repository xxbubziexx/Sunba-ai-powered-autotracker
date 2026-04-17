"""
Splits finetuning/train into train (90%) and valid (10%).
Run once before training: python split_dataset.py
"""

import random
import shutil
from pathlib import Path

TRAIN_IMAGES = Path("finetuning/train/images")
TRAIN_LABELS = Path("finetuning/train/labels")
VALID_IMAGES = Path("finetuning/valid/images")
VALID_LABELS = Path("finetuning/valid/labels")
SPLIT        = 0.10   # fraction to move to valid

images = list(TRAIN_IMAGES.glob("*.jpg")) + list(TRAIN_IMAGES.glob("*.png"))
random.shuffle(images)

n_valid = max(1, int(len(images) * SPLIT))
val_set = images[:n_valid]

VALID_IMAGES.mkdir(parents=True, exist_ok=True)
VALID_LABELS.mkdir(parents=True, exist_ok=True)

for img in val_set:
    lbl = TRAIN_LABELS / img.with_suffix(".txt").name
    shutil.move(str(img), VALID_IMAGES / img.name)
    if lbl.exists():
        shutil.move(str(lbl), VALID_LABELS / lbl.name)

print(f"Total: {len(images)} | Train: {len(images) - n_valid} | Valid: {n_valid}")
