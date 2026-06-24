# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
Source dataset layout (already on disk):
    datasets/
        train/                 # 930 images + train.json (COCO-format)
        val/                   # 234 images + test.json  (COCO-format)

The Roboflow loader inside rfdetr expects the COCO annotations to live at
``train/_annotations.coco.json`` and ``valid/_annotations.coco.json``. This
script stages a small directory of symlinks into that layout, then runs
``RFDETRMedium.train`` with hyperparameters tuned.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DATASET = REPO_ROOT / "datasets"
STAGED_DATASET = REPO_ROOT / "datasets_rf"
OUTPUT_DIR = REPO_ROOT / "output_medium"

# Cache the rfdetr pretrained checkpoint in the project root itself.
os.environ["RF_HOME"] = str(REPO_ROOT)

import torch  # noqa: E402

from rfdetr import RFDETRMedium  # noqa: E402


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink src -> dst, falling back to copy when symlinks are not permitted."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def stage_roboflow_layout() -> Path:
    """Create datasets_rf/{train,valid}/_annotations.coco.json layout via symlinks."""
    src_train_dir = SRC_DATASET / "train"
    src_val_dir = SRC_DATASET / "val"
    src_train_json = src_train_dir / "train.json"
    src_val_json = src_val_dir / "test.json"

    for path in (src_train_dir, src_val_dir, src_train_json, src_val_json):
        if not path.exists():
            sys.exit(f"Missing dataset path: {path}")

    STAGED_DATASET.mkdir(parents=True, exist_ok=True)
    staged_train = STAGED_DATASET / "train"
    staged_valid = STAGED_DATASET / "valid"

    _link_or_copy(src_train_dir, staged_train)
    _link_or_copy(src_val_dir, staged_valid)

    # The image folders already exist via the directory symlink; the COCO
    # loader looks for the annotation JSON at a fixed filename next to them.
    train_ann = staged_train / "_annotations.coco.json"
    valid_ann = staged_valid / "_annotations.coco.json"
    if not train_ann.exists():
        train_ann.symlink_to(src_train_json.resolve())
    if not valid_ann.exists():
        valid_ann.symlink_to(src_val_json.resolve())
    return STAGED_DATASET


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    dataset_dir = stage_roboflow_layout()
    device = pick_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = RFDETRMedium()
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(OUTPUT_DIR),
        device=device,
        epochs=50,
        batch_size=2,
        grad_accum_steps=8,
        lr=1e-4,
        lr_encoder=1.5e-4,
        warmup_epochs=1.0,
        weight_decay=1e-4,
        num_workers=2,
        checkpoint_interval=5,
        early_stopping=True,
        early_stopping_patience=10,
        tensorboard=True,
        wandb=False,
        # Lighter aug stack — multi-scale crops are heavy on a Mac CPU/MPS.
        multi_scale=False,
        expanded_scales=False,
        square_resize_div_64=True,
    )


if __name__ == "__main__":
    main()
