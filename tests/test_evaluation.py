"""
Unit tests for evaluation metrics, confusion matrices, and edge case handling.
"""

from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from classifier_models import CustomTrafficCNN, save_checkpoint
from dataset_utils import create_synthetic_sample_dataset
from evaluate_classifier import evaluate_classifier


def test_perfect_classification_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    assert acc == 1.0
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0


def test_zero_division_edge_case():
    # Class 2 is never predicted
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 0, 0, 1, 0])

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    assert prec >= 0.0
    assert rec >= 0.0
    assert f1 >= 0.0


def test_confusion_matrix_dimensions():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 2, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # class 0 predicted as 0
    assert cm[1, 2] == 1  # class 1 predicted as 2
    assert cm[2, 1] == 1  # class 2 predicted as 1


def test_evaluate_classifier_pipeline(tmp_path):
    dataset_dir = tmp_path / "data"
    create_synthetic_sample_dataset(dataset_dir, samples_per_class=3, seed=42)

    class_names = ["auto_rickshaw", "bicycle", "bus", "car", "motorcycle", "scooter", "truck"]
    model = CustomTrafficCNN(num_classes=len(class_names))

    cp_path = tmp_path / "test_eval_model.pth"
    save_checkpoint(
        model=model,
        architecture="custom_cnn",
        class_names=class_names,
        filepath=cp_path,
        image_size=224,
    )

    out_dir = tmp_path / "results"
    results = evaluate_classifier(
        checkpoint_path=cp_path,
        test_dir=dataset_dir / "test",
        batch_size=4,
        device_name="cpu",
        output_dir=out_dir,
    )

    assert "accuracy" in results
    assert "macro_f1" in results
    assert "confusion_matrix" in results
    assert len(results["per_class_metrics"]) == len(class_names)
    assert Path(results["confusion_matrix_plot"]).is_file()
    assert Path(results["per_class_csv"]).is_file()
