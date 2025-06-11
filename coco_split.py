# -*- coding: utf-8 -*-
"""
Created on Sun May 18 09:07:25 2025

@author: JS
"""

import os
import shutil
import random
from pathlib import Path
from typing import List

def split_coco_pairs(
    image_dirs: List[Path],
    label_dirs: List[Path],
    output_dir: Path,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    test_frac: float  = 0.15,
    seed: int = 42
):
    """
    - image_dirs:   list of existing image folders to merge (e.g. [train2017, val2017])
    - label_dirs:   list of existing label folders to merge
    - output_dir:   base path where images/{train,val,test} and labels/{train,val,test} will be created
    - train/val/test fractions must sum to 1.0
    - seed:         for reproducibility
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1."

    # 1) gather all files and map by basename
    image_map = {}
    for d in image_dirs:
        for img in d.iterdir():
            if img.is_file():
                image_map[img.stem] = img

    label_map = {}
    for d in label_dirs:
        for lbl in d.iterdir():
            if lbl.is_file():
                label_map[lbl.stem] = lbl

    # 2) keep only names present in both
    common_names = list(set(image_map) & set(label_map))
    print(f"Found {len(common_names)} pairs out of {len(image_map)} images & {len(label_map)} labels.")

    # 3) shuffle & split indices
    random.seed(seed)
    random.shuffle(common_names)
    N = len(common_names)
    n_train = int(train_frac * N)
    n_val   = int(val_frac   * N)
    # ensure all used
    n_test  = N - n_train - n_val

    splits = {
        'train': common_names[:n_train],
        'val':   common_names[n_train:n_train+n_val],
        'test':  common_names[n_train+n_val:]
    }

    # 4) create output folders
    for split in splits:
        for sub in ('images', 'labels'):
            (output_dir / sub / split).mkdir(parents=True, exist_ok=True)

    # 5) copy files
    for split, names in splits.items():
        for name in names:
            shutil.copy(image_map[name], output_dir / 'images' / split / image_map[name].name)
            shutil.copy(label_map[name], output_dir / 'labels' / split / label_map[name].name)

    print("Done.")

if __name__=="__main__":
    from pathlib import Path

# your current folders:
image_dirs = [Path("dataset/COCO/images/train2017"),
              Path("dataset/COCO/images/val2017")]
label_dirs = [Path("dataset/COCO/labels/train2017"),
              Path("dataset/COCO/labels/val2017")]

# where to write the new split:
output_dir = Path("dataset/COCO")

split_coco_pairs(image_dirs, label_dirs, output_dir)
