"""
Dataset Utilities for Smart Traffic Monitoring System.
Handles validation, corrupt image detection, class imbalance computation,
transforms, PyTorch data loading, and synthetic sample dataset generation.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from config import (
    CLASS_COLORS_BGR,
    DEFAULT_IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    SUPPORTED_IMAGE_EXTS,
    VEHICLE_CLASSES,
)


def is_valid_image(filepath: Union[str, Path]) -> bool:
    """
    Check if an image file exists, has a supported extension, and can be decoded by PIL without corruption.
    """
    path = Path(filepath)
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        # Verify only reads header; load actually decodes pixels
        with Image.open(path) as img:
            img.load()
            if img.mode not in ("RGB", "L", "RGBA"):
                return False
        return True
    except Exception:
        return False


def validate_dataset(dataset_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate dataset directory structure, class consistency, corrupt images, and class imbalance.
    Expected structure:
        <dataset_dir>/train/<class_name>/*.jpg
        <dataset_dir>/val/<class_name>/*.jpg
        <dataset_dir>/test/<class_name>/*.jpg
    Returns:
        Structured validation report.
    """
    path = Path(dataset_dir)
    report: Dict[str, Any] = {
        "is_valid": True,
        "dataset_path": str(path),
        "splits_found": [],
        "missing_splits": [],
        "classes_by_split": {},
        "counts_by_split": {},
        "corrupt_images": [],
        "unsupported_files": [],
        "total_valid_images": 0,
        "class_imbalance": {},
        "errors": [],
        "warnings": [],
    }

    if not path.is_dir():
        report["is_valid"] = False
        report["errors"].append(f"Dataset root directory does not exist: {path}")
        return report

    expected_splits = ["train", "val", "test"]
    for split in expected_splits:
        split_dir = path / split
        if split_dir.is_dir():
            report["splits_found"].append(split)
        else:
            report["missing_splits"].append(split)

    if not report["splits_found"]:
        # Check if direct class folders exist (raw un-split dataset)
        class_dirs = [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if class_dirs:
            report["warnings"].append(
                f"Directory appears to be an un-split dataset with classes: {class_dirs}. "
                "Use 'split_dataset.py' to partition it into train/val/test splits."
            )
        else:
            report["is_valid"] = False
            report["errors"].append(f"No train/val/test splits found in {path}")
        return report

    reference_classes = None
    all_classes_set = set()

    for split in report["splits_found"]:
        split_dir = path / split
        class_folders = sorted(
            [d.name for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )
        report["classes_by_split"][split] = class_folders
        all_classes_set.update(class_folders)

        if reference_classes is None:
            reference_classes = class_folders
        elif class_folders != reference_classes:
            report["warnings"].append(
                f"Class folder mismatch in '{split}' split compared to other splits: {class_folders} vs {reference_classes}"
            )

        # Count images and check for corrupt files
        split_counts = {}
        for c in class_folders:
            c_dir = split_dir / c
            valid_count = 0
            for item in c_dir.iterdir():
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext not in SUPPORTED_IMAGE_EXTS:
                        report["unsupported_files"].append(str(item))
                        continue
                    if is_valid_image(item):
                        valid_count += 1
                    else:
                        report["corrupt_images"].append(str(item))
            split_counts[c] = valid_count
            report["total_valid_images"] += valid_count

        report["counts_by_split"][split] = split_counts

    # Check for empty classes
    for split, counts in report["counts_by_split"].items():
        for c, count in counts.items():
            if count == 0:
                report["warnings"].append(f"Class '{c}' in '{split}' split has 0 valid images.")

    # Compute class imbalance for train split if present
    if "train" in report["counts_by_split"]:
        train_counts = report["counts_by_split"]["train"]
        if train_counts and sum(train_counts.values()) > 0:
            total_train = sum(train_counts.values())
            max_c = max(train_counts.values())
            min_c = min(c for c in train_counts.values() if c > 0) if any(c > 0 for c in train_counts.values()) else 1
            imbalance_ratio = round(max_c / max(min_c, 1), 2)
            percentages = {
                k: round((v / total_train) * 100, 2) for k, v in train_counts.items()
            }
            report["class_imbalance"] = {
                "max_count": max_c,
                "min_count": min_c,
                "imbalance_ratio": imbalance_ratio,
                "percentages": percentages,
                "severity": (
                    "Severe" if imbalance_ratio > 5.0
                    else "Moderate" if imbalance_ratio > 2.0
                    else "Balanced"
                ),
            }

    if report["corrupt_images"]:
        report["warnings"].append(
            f"Detected {len(report['corrupt_images'])} corrupt or unreadable image(s)."
        )

    # Missing splits error
    if "train" not in report["splits_found"]:
        report["is_valid"] = False
        report["errors"].append("Missing required 'train' split folder.")

    return report


def get_data_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    is_train: bool = True,
    augment: bool = True,
) -> transforms.Compose:
    """
    Get PyTorch transforms for training, validation, or test.
    Augmentation is applied ONLY when is_train=True and augment=True.
    """
    if is_train and augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ]
        )


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    Compute inverse class frequency weights to handle class imbalance in CrossEntropyLoss:
    weight[c] = N / (C * N_c)
    """
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    total_samples = len(targets)
    num_classes = len(dataset.classes)

    weights = []
    for idx in range(num_classes):
        count = class_counts.get(idx, 0)
        if count > 0:
            w = total_samples / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


def create_synthetic_sample_dataset(
    output_dir: Union[str, Path],
    samples_per_class: int = 12,
    image_size: Tuple[int, int] = (224, 224),
    seed: int = 42,
) -> Path:
    """
    Generate synthetic sample images for all 7 vehicle classes with distinct visual features,
    geometries, and labels. Creates train, val, and test splits.
    Allows out-of-the-box verification of training, evaluation, comparison, and inference
    without requiring manual download of external datasets.
    """
    out_path = Path(output_dir)
    np.random.seed(seed)

    splits = {
        "train": int(samples_per_class * 0.70),
        "val": int(samples_per_class * 0.15),
        "test": max(1, samples_per_class - int(samples_per_class * 0.70) - int(samples_per_class * 0.15)),
    }

    # Ensure min 1 sample in each split
    for s in splits:
        if splits[s] < 1:
            splits[s] = 1

    for split_name, count in splits.items():
        for cls_name in VEHICLE_CLASSES:
            cls_dir = out_path / split_name / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            color_bgr = CLASS_COLORS_BGR.get(cls_name, (128, 128, 128))
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

            for idx in range(count):
                img_file = cls_dir / f"{cls_name}_{split_name}_{idx:03d}.png"
                if img_file.exists():
                    continue

                # Create synthetic canvas with realistic background gradient
                canvas = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
                # Road/asphalt background with subtle variation
                bg_intensity = np.random.randint(40, 75)
                canvas[:, :] = [bg_intensity, bg_intensity, bg_intensity]

                # Draw road markings
                canvas[image_size[1] // 2 - 4 : image_size[1] // 2 + 4, :] = [200, 200, 200]

                pil_img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(pil_img)

                # Draw vehicle body geometry matching vehicle type
                w, h = image_size
                box_margin = np.random.randint(25, 45)

                if cls_name == "auto_rickshaw":
                    # Distinctive yellow-green cabin
                    cabin_box = [box_margin, box_margin + 20, w - box_margin, h - box_margin]
                    draw.rectangle(cabin_box, fill=color_rgb, outline=(255, 255, 255), width=3)
                    # 3 wheels (1 front, 2 rear)
                    draw.ellipse([w // 2 - 12, box_margin, w // 2 + 12, box_margin + 24], fill=(20, 20, 20))
                    draw.ellipse([box_margin, h - box_margin - 20, box_margin + 24, h - box_margin + 4], fill=(20, 20, 20))
                    draw.ellipse([w - box_margin - 24, h - box_margin - 20, w - box_margin, h - box_margin + 4], fill=(20, 20, 20))
                elif cls_name in ("motorcycle", "scooter"):
                    # 2 wheels and elongated frame
                    draw.ellipse([w // 2 - 14, box_margin + 10, w // 2 + 14, box_margin + 38], fill=(15, 15, 15))
                    draw.ellipse([w // 2 - 14, h - box_margin - 38, w // 2 + 14, h - box_margin - 10], fill=(15, 15, 15))
                    draw.line([w // 2, box_margin + 24, w // 2, h - box_margin - 24], fill=color_rgb, width=12)
                elif cls_name == "bicycle":
                    # Thin frame with two circles
                    draw.ellipse([w // 2 - 10, box_margin + 10, w // 2 + 10, box_margin + 30], outline=(200, 200, 200), width=4)
                    draw.ellipse([w // 2 - 10, h - box_margin - 30, w // 2 + 10, h - box_margin - 10], outline=(200, 200, 200), width=4)
                    draw.line([w // 2, box_margin + 20, w // 2, h - box_margin - 20], fill=color_rgb, width=4)
                elif cls_name in ("bus", "truck"):
                    # Large rectangular body
                    draw.rectangle([box_margin - 10, box_margin, w - box_margin + 10, h - box_margin], fill=color_rgb, outline=(255, 255, 255), width=3)
                    # Windshield
                    draw.rectangle([box_margin, box_margin + 10, w - box_margin, box_margin + 40], fill=(100, 150, 220))
                else:  # car
                    # Car profile
                    draw.rectangle([box_margin, box_margin + 15, w - box_margin, h - box_margin - 15], fill=color_rgb, outline=(255, 255, 255), width=3)
                    draw.rectangle([box_margin + 20, box_margin + 30, w - box_margin - 20, h - box_margin - 30], fill=(40, 60, 90))

                # Text label watermark
                draw.text(
                    (15, h - 25),
                    f"{cls_name} #{idx}",
                    fill=(255, 255, 255),
                )

                pil_img.save(img_file)

    return out_path
