"""
Model Evaluation Module for Vehicle Classification.
Computes overall accuracy, macro/weighted precision, recall, F1-score,
per-class breakdown, and confusion matrix heatmap on a held-out test split.
Saves structured JSON and CSV reports and updates SQLite experiment records.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from classifier_models import load_checkpoint
from config import RESULTS_DIR, get_device
from database import get_experiments, update_experiment_test_metrics
from dataset_utils import get_data_transforms
from utils import save_json, setup_logger

logger = setup_logger("evaluate_classifier")

METRIC_DISTINCTION_NOTE = """
[CRITICAL CONCEPTUAL DISTINCTION: CLASSIFICATION vs. DETECTION METRICS]
1. Classification Metrics (Accuracy, Precision, Recall, Macro-F1):
   - Evaluated on CROPPED vehicle image patches.
   - Measures: 'Given a pre-cropped vehicle image, is the vehicle category identified correctly?'
   - Does NOT assess whether the vehicle's position, scale, or boundaries were found in a complex street scene.

2. Detection Metrics (mAP@50, mAP@50-95, Recall, IoU):
   - Evaluated on FULL road scene images with ground-truth bounding box coordinates.
   - Measures: 'Did the model locate all vehicles with sufficient Intersection-over-Union (IoU) overlap AND classify each box correctly?'
   - To benchmark YOLO detector performance, standard YOLO format labels (class x_center y_center width height) are required.
"""


def evaluate_classifier(
    checkpoint_path: Union[str, Path],
    test_dir: Union[str, Path],
    batch_size: int = 32,
    device_name: str = "auto",
    output_dir: Union[str, Path] = RESULTS_DIR,
) -> Dict[str, Any]:
    """
    Run full quantitative evaluation on held-out test dataset.
    """
    checkpoint_file = Path(checkpoint_path).resolve()
    test_path = Path(test_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    device = get_device(device_name)
    logger.info(f"Loading checkpoint {checkpoint_file.name} onto {device}...")

    model, meta = load_checkpoint(checkpoint_file, device=device)
    arch = meta.get("architecture", "custom_cnn")
    expected_classes = meta.get("class_names", [])
    image_size = meta.get("image_size", 224)

    test_transform = get_data_transforms(image_size=image_size, is_train=False, augment=False)
    test_dataset = datasets.ImageFolder(str(test_path), transform=test_transform)
    test_classes = test_dataset.classes

    if test_classes != expected_classes:
        logger.warning(
            f"Class list in test dataset {test_classes} differs from model checkpoint {expected_classes}. "
            "Metrics will be aligned to model's index space."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[List[float]] = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_scores.extend(probs.cpu().numpy().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Calculate overall metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )

    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(
            y_true_arr,
            y_pred_arr,
            labels=list(range(len(expected_classes))),
            average=None,
            zero_division=0,
        )
    )

    per_class_list = []
    for idx, cname in enumerate(expected_classes):
        per_class_list.append(
            {
                "class_name": cname,
                "precision": round(float(prec_per_class[idx]), 4),
                "recall": round(float(rec_per_class[idx]), 4),
                "f1_score": round(float(f1_per_class[idx]), 4),
                "support": int(support_per_class[idx]),
            }
        )

    # Confusion Matrix
    cm = confusion_matrix(
        y_true_arr, y_pred_arr, labels=list(range(len(expected_classes)))
    )

    # Normalized Confusion Matrix
    cm_norm = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            cm_norm[i] = cm[i] / row_sums[i]

    # Save Confusion Matrix Plot
    cm_fig_path = out_path / f"{arch}_confusion_matrix.png"
    plt.figure(figsize=(9, 7))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{arch} - Normalized Confusion Matrix (Test Split)", fontsize=13)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(expected_classes))
    plt.xticks(tick_marks, expected_classes, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, expected_classes, fontsize=10)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            val_norm = cm_norm[i, j]
            plt.text(
                j,
                i,
                f"{val}\n({val_norm:.2f})",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val_norm > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(cm_fig_path, dpi=160)
    plt.close()

    # Save Per-Class CSV
    per_class_df = pd.DataFrame(per_class_list)
    per_class_csv = out_path / f"{arch}_per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv, index=False)

    # Save JSON Report
    results = {
        "architecture": arch,
        "checkpoint_path": str(checkpoint_file),
        "test_dataset_path": str(test_path),
        "total_test_samples": len(y_true),
        "accuracy": round(acc, 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class_metrics": per_class_list,
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": np.round(cm_norm, 4).tolist(),
        "confusion_matrix_plot": str(cm_fig_path),
        "per_class_csv": str(per_class_csv),
        "evaluation_notes": METRIC_DISTINCTION_NOTE.strip(),
    }

    json_report_path = out_path / f"{arch}_test_metrics.json"
    save_json(results, json_report_path)

    # Link back to experiment in SQLite database if found
    experiments = get_experiments()
    for exp in experiments:
        if exp.get("checkpoint_path") == str(checkpoint_file):
            update_experiment_test_metrics(
                experiment_id=exp["id"],
                test_accuracy=round(acc, 4),
                test_macro_f1=round(float(f1_macro), 4),
                metrics_json_path=str(json_report_path),
            )
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate vehicle classifier on test dataset.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to held-out test folder")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save evaluation reports")

    args = parser.parse_args()

    try:
        report = evaluate_classifier(
            checkpoint_path=args.checkpoint,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            device_name=args.device,
            output_dir=args.output_dir,
        )
        print("\n" + "=" * 65)
        print(f"  Evaluation Report: {report['architecture'].upper()}")
        print("=" * 65)
        print(f"Test Accuracy:    {report['accuracy']*100:.2f}%")
        print(f"Macro Precision:  {report['macro_precision']*100:.2f}%")
        print(f"Macro Recall:     {report['macro_recall']*100:.2f}%")
        print(f"Macro F1-Score:   {report['macro_f1']*100:.2f}%")
        print(f"Total Test Items: {report['total_test_samples']}")
        print(f"Report JSON:      {report['checkpoint_path']}")
        print(f"Confusion Matrix: {report['confusion_matrix_plot']}")
        print("\nPer-Class Breakdown:")
        for c in report["per_class_metrics"]:
            print(f"  {c['class_name']:<15} | Prec: {c['precision']*100:5.1f}% | Rec: {c['recall']*100:5.1f}% | F1: {c['f1_score']*100:5.1f}% | Support: {c['support']}")
        print(METRIC_DISTINCTION_NOTE)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
