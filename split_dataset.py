"""
Dataset Splitting Module for Vehicle Classification.
Partitions a raw folder of class-named images into train, val, and test splits.
Uses stratified sampling with random seeds for full reproducibility.
Defaults to COPYING images to prevent data destruction.
Includes explicit data leakage warnings for sequential video frames.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from config import SUPPORTED_IMAGE_EXTS
from dataset_utils import is_valid_image
from utils import set_seed, setup_logger

logger = setup_logger("split_dataset")

DATA_LEAKAGE_WARNING = """
[DATA LEAKAGE WARNING]
If your dataset consists of sequential video frames extracted from road recordings:
DO NOT randomly split adjacent frames across train and test sets!
Consecutive frames have nearly identical backgrounds, lighting, and vehicle angles,
which artificially inflates validation/test metrics.
Instead, split by complete recording sessions, distinct dates, or camera viewpoints.
"""


def split_dataset(
    source_dir: Path,
    dest_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    move_files: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Split a class-structured raw dataset into train/val/test splits.
    Args:
        source_dir: Path containing subfolders for each class
        dest_dir: Target path where train/, val/, test/ will be created
        train_ratio: Proportion for training set (e.g., 0.70)
        val_ratio: Proportion for validation set (e.g., 0.15)
        test_ratio: Proportion for testing set (e.g., 0.15)
        seed: Random seed for reproducibility
        move_files: If True, moves files; if False (default), copies files safely
    Returns:
        Summary dictionary with counts per split and per class.
    """
    set_seed(seed)
    source_path = Path(source_dir).resolve()
    dest_path = Path(dest_dir).resolve()

    if not source_path.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not (0.99 <= ratio_sum <= 1.01):
        raise ValueError(
            f"Split ratios must sum to 1.0. Current sum: {ratio_sum:.4f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )

    print(DATA_LEAKAGE_WARNING)

    # Discover classes
    class_dirs = [
        d for d in source_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    if not class_dirs:
        raise ValueError(
            f"No class subfolders found inside {source_path}. "
            f"Expected structure: {source_path}/<class_name>/*.jpg"
        )

    summary: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for class_dir in class_dirs:
        class_name = class_dir.name
        # Gather valid images
        all_images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS and is_valid_image(f)
        ]

        if not all_images:
            logger.warning(f"Class '{class_name}' has 0 valid images. Skipping.")
            continue

        # Deterministic shuffle
        random.seed(seed)
        shuffled = sorted(all_images)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Ensure at least 1 image per split if total >= 3
        if n_total >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
        n_test = n_total - n_train - n_val

        splits: Dict[str, List[Path]] = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }

        # Perform copying/moving
        action_fn = shutil.move if move_files else shutil.copy2
        action_name = "Moved" if move_files else "Copied"

        for split_name, img_list in splits.items():
            target_class_dir = dest_path / split_name / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for img_file in img_list:
                target_file = target_class_dir / img_file.name
                action_fn(img_file, target_file)

            summary[split_name][class_name] = len(img_list)

        logger.info(
            f"Class '{class_name}' ({n_total} images) -> "
            f"train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])} ({action_name})"
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Stratified dataset splitter for vehicle classification."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw",
        help="Source directory containing class subfolders",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default="dataset",
        help="Destination directory to save train/val/test splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion for training split (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion for validation split (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion for test split (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split (default: 42)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        default=False,
        help="Move original files instead of copying (WARNING: Destructive)",
    )

    args = parser.parse_args()

    try:
        summary = split_dataset(
            source_dir=Path(args.source),
            dest_dir=Path(args.destination),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            move_files=args.move,
        )
        print("\n=== Dataset Partition Summary ===")
        for split, class_counts in summary.items():
            total_in_split = sum(class_counts.values())
            print(f"[{split.upper()}] Total: {total_in_split}")
            for c, cnt in class_counts.items():
                print(f"  - {c}: {cnt}")
    except Exception as e:
        logger.error(f"Dataset split failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
