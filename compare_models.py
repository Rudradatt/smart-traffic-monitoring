"""
Model Comparison and Benchmarking Module.
Compares Custom CNN, ResNet-18, and MobileNetV3-Small on identical test splits.
Measures:
- Parameter count (Total & Trainable)
- Checkpoint file size (MB)
- Inference latency (ms per image) & throughput (FPS)
- Accuracy, Macro-Precision, Macro-Recall, Macro-F1
- Hardware device metadata
Exports CSV, JSON, and comparison charts, recording all findings in SQLite.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from classifier_models import load_checkpoint
from config import RESULTS_DIR, get_device
from database import log_model_comparison
from evaluate_classifier import evaluate_classifier
from utils import (
    calculate_model_parameters,
    get_device_info,
    get_file_size_mb,
    save_json,
    setup_logger,
)

logger = setup_logger("compare_models")


def measure_inference_speed(
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> Dict[str, float]:
    """
    Measure model inference latency (ms) and throughput (FPS) using synthetic single-image inputs.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(benchmark_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    total_time = time.time() - start_time
    avg_latency_ms = (total_time / benchmark_runs) * 1000.0
    throughput_fps = benchmark_runs / total_time

    return {
        "latency_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(throughput_fps, 2),
    }


def compare_models(
    checkpoints: List[Union[str, Path]],
    test_dir: Union[str, Path],
    output_dir: Union[str, Path] = RESULTS_DIR,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    Run comprehensive side-by-side benchmarking of multiple model checkpoints.
    """
    test_path = Path(test_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    device = get_device(device_name)
    dev_info = get_device_info()

    if not checkpoints:
        raise ValueError("At least one checkpoint path must be provided for comparison.")

    records = []
    logger.info(f"Comparing {len(checkpoints)} models on {dev_info['device_name']}...")

    for cp_path in checkpoints:
        path = Path(cp_path).resolve()
        if not path.is_file():
            logger.warning(f"Checkpoint not found, skipping: {path}")
            continue

        logger.info(f"Benchmarking checkpoint: {path.name}...")
        model, meta = load_checkpoint(path, device=device)
        arch = meta.get("architecture", path.stem)
        image_size = meta.get("image_size", 224)

        # 1. Parameter counts
        param_info = calculate_model_parameters(model)
        size_mb = get_file_size_mb(path)

        # 2. Latency & throughput
        speed_info = measure_inference_speed(
            model=model,
            image_size=image_size,
            device=device,
        )

        # 3. Test set evaluation
        eval_metrics = evaluate_classifier(
            checkpoint_path=path,
            test_dir=test_path,
            batch_size=32,
            device_name=device_name,
            output_dir=out_path,
        )

        record = {
            "checkpoint": path.name,
            "architecture": arch,
            "total_params": param_info["total_params"],
            "trainable_params": param_info["trainable_params"],
            "size_mb": size_mb,
            "latency_ms": speed_info["latency_ms"],
            "throughput_fps": speed_info["throughput_fps"],
            "test_accuracy": eval_metrics["accuracy"],
            "macro_f1": eval_metrics["macro_f1"],
            "macro_precision": eval_metrics["macro_precision"],
            "macro_recall": eval_metrics["macro_recall"],
            "device": dev_info["device_name"],
        }
        records.append(record)

    if not records:
        raise RuntimeError("No valid checkpoints could be evaluated.")

    comp_df = pd.DataFrame(records)
    csv_path = out_path / "models_comparison.csv"
    comp_df.to_csv(csv_path, index=False)

    # Plot Comparison Charts
    fig_path = out_path / "models_comparison_chart.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Test Accuracy
    axes[0, 0].bar(comp_df["architecture"], comp_df["test_accuracy"] * 100, color="#34C759")
    axes[0, 0].set_title("Test Accuracy (%)", fontsize=12)
    axes[0, 0].set_ylabel("Accuracy %")
    axes[0, 0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(comp_df["test_accuracy"]):
        axes[0, 0].text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontweight="bold")

    # 2. Macro F1
    axes[0, 1].bar(comp_df["architecture"], comp_df["macro_f1"] * 100, color="#007AFF")
    axes[0, 1].set_title("Macro F1-Score (%)", fontsize=12)
    axes[0, 1].set_ylabel("F1 %")
    axes[0, 1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(comp_df["macro_f1"]):
        axes[0, 1].text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontweight="bold")

    # 3. Latency (ms)
    axes[1, 0].bar(comp_df["architecture"], comp_df["latency_ms"], color="#FF9500")
    axes[1, 0].set_title("Inference Latency (ms/img, lower is better)", fontsize=12)
    axes[1, 0].set_ylabel("Milliseconds")
    axes[1, 0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(comp_df["latency_ms"]):
        axes[1, 0].text(i, v + 0.5, f"{v:.1f}ms", ha="center")

    # 4. Total Parameters (Millions)
    params_m = comp_df["total_params"] / 1e6
    axes[1, 1].bar(comp_df["architecture"], params_m, color="#AF52DE")
    axes[1, 1].set_title("Total Parameters (Millions)", fontsize=12)
    axes[1, 1].set_ylabel("Parameters (M)")
    axes[1, 1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(params_m):
        axes[1, 1].text(i, v + 0.1, f"{v:.2f}M", ha="center")

    plt.suptitle(f"Model Comparison Benchmark ({dev_info['device_name']})", fontsize=15, y=0.99)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    summary_result = {
        "hardware": dev_info,
        "models_count": len(records),
        "comparison_table": records,
        "comparison_csv": str(csv_path),
        "comparison_plot": str(fig_path),
    }

    json_path = out_path / "models_comparison.json"
    save_json(summary_result, json_path)

    # Record in SQLite
    log_model_comparison(
        comparison_name=f"Comparison_{int(time.time())}",
        models_evaluated=[str(c) for c in checkpoints],
        summary=summary_result,
    )

    return summary_result


def main():
    parser = argparse.ArgumentParser(description="Compare multiple trained vehicle classifiers.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of model checkpoint paths (.pth files) to compare",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="dataset/test",
        help="Held-out test dataset directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for reports",
    )

    args = parser.parse_args()

    try:
        res = compare_models(
            checkpoints=args.checkpoints,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            device_name=args.device,
        )
        print("\n" + "=" * 70)
        print("  Model Comparison Benchmark Results")
        print("=" * 70)
        df = pd.DataFrame(res["comparison_table"])
        cols = ["architecture", "size_mb", "latency_ms", "throughput_fps", "test_accuracy", "macro_f1"]
        print(df[cols].to_string(index=False))
        print(f"\nSaved CSV:  {res['comparison_csv']}")
        print(f"Saved Chart: {res['comparison_plot']}")
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
