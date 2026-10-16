"""
Unit tests for dataset utilities and splitting logic.
"""

import tempfile
from pathlib import Path
import pytest
from PIL import Image

from dataset_utils import (
    create_synthetic_sample_dataset,
    get_data_transforms,
    is_valid_image,
    validate_dataset,
)
from split_dataset import split_dataset
from config import VEHICLE_CLASSES


def test_is_valid_image_with_valid_image(tmp_path):
    img_path = tmp_path / "valid.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(img_path)

    assert is_valid_image(img_path) is True


def test_is_valid_image_with_corrupt_file(tmp_path):
    corrupt_path = tmp_path / "corrupt.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_IMAGE_HEADER_OR_DATA")

    assert is_valid_image(corrupt_path) is False


def test_is_valid_image_with_unsupported_ext(tmp_path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("hello")

    assert is_valid_image(txt_path) is False


def test_validate_dataset_missing_dir(tmp_path):
    non_existent = tmp_path / "does_not_exist"
    report = validate_dataset(non_existent)
    assert report["is_valid"] is False
    assert len(report["errors"]) > 0


def test_create_synthetic_dataset_and_validate(tmp_path):
    out_dir = tmp_path / "synthetic_dataset"
    create_synthetic_sample_dataset(out_dir, samples_per_class=4, seed=42)

    report = validate_dataset(out_dir)
    assert report["is_valid"] is True
    assert "train" in report["splits_found"]
    assert "val" in report["splits_found"]
    assert "test" in report["splits_found"]
    assert report["total_valid_images"] > 0

    # Ensure all 7 classes exist in the train split
    train_classes = report["classes_by_split"]["train"]
    for c in VEHICLE_CLASSES:
        assert c in train_classes


def test_split_dataset_reproducibility_and_ratios(tmp_path):
    # Create raw directory with two classes
    raw_dir = tmp_path / "raw"
    dest_dir1 = tmp_path / "split1"
    dest_dir2 = tmp_path / "split2"

    for c in ["car", "bus"]:
        c_dir = raw_dir / c
        c_dir.mkdir(parents=True)
        for i in range(10):
            img = Image.new("RGB", (64, 64), color=(i * 10, i * 10, i * 10))
            img.save(c_dir / f"img_{i:02d}.png")

    # Split with same seed
    res1 = split_dataset(raw_dir, dest_dir1, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=123)
    res2 = split_dataset(raw_dir, dest_dir2, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=123)

    assert res1 == res2
    assert res1["train"]["car"] == 7
    assert res1["val"]["car"] == 2
    assert res1["test"]["car"] == 1


def test_split_dataset_invalid_ratios(tmp_path):
    raw_dir = tmp_path / "raw"
    dest_dir = tmp_path / "dest"
    raw_dir.mkdir()

    with pytest.raises(ValueError):
        split_dataset(raw_dir, dest_dir, train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)


def test_transforms_construction():
    t_train = get_data_transforms(image_size=224, is_train=True, augment=True)
    t_val = get_data_transforms(image_size=224, is_train=False, augment=False)

    assert len(t_train.transforms) > len(t_val.transforms)
