"""
Model Training Script for Vehicle Classification.
Supports:
- Custom CNN from scratch
- ResNet-18 Transfer Learning
- MobileNetV3-Small Transfer Learning
Includes class weight compensation, early stopping, LR scheduling,
checkpoint saving, training curves plotting, and SQLite experiment logging.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from classifier_models import build_classifier, save_checkpoint
from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    MODELS_DIR,
    RESULTS_DIR,
    get_device,
)
from database import log_experiment
from dataset_utils import (
    compute_class_weights,
    get_data_transforms,
    validate_dataset,
)
from utils import format_time, set_seed, setup_logger

logger = setup_logger("train_classifier")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for a single epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation split. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc


def train_classifier(
    data_dir: Path,
    arch: str = "custom_cnn",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    device_name: str = "auto",
    patience: int = DEFAULT_PATIENCE,
    use_class_weights: bool = True,
    save_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """
    Complete model training pipeline.
    """
    set_seed(seed)
    device = get_device(device_name)
    data_path = Path(data_dir).resolve()
    save_path = Path(save_dir).resolve()
    results_path = Path(results_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target Device: {device} | Architecture: {arch}")
    logger.info(f"Checking dataset structure in: {data_path}")

    # Validate dataset
    val_report = validate_dataset(data_path)
    if not val_report["is_valid"]:
        raise ValueError(f"Dataset validation failed: {val_report['errors']}")

    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected 'train' and 'val' subdirectories inside {data_path}."
        )

    # Transforms (train gets augmentation, val gets clean resize)
    train_transform = get_data_transforms(image_size=image_size, is_train=True, augment=True)
    val_transform = get_data_transforms(image_size=image_size, is_train=False, augment=False)

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    logger.info(f"Inferred {num_classes} classes: {class_names}")

    if num_classes < 2:
        raise ValueError(f"Need at least 2 classes for classification. Found {num_classes}.")

    drop_last = (len(train_dataset) > batch_size and len(train_dataset) % batch_size == 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,  # Safe default on Windows
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model instantiation
    model = build_classifier(
        arch=arch,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=False,
    )
    model.to(device)

    # Class weights for imbalanced data
    if use_class_weights:
        weights = compute_class_weights(train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        logger.info(f"Using class weights: {[round(w.item(), 3) for w in weights]}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_checkpoint_path = save_path / f"best_{arch}.pth"
    start_total_time = time.time()

    print("\n" + "=" * 65)
    print(f"  Training {arch.upper()} on {device.type.upper()} ({num_classes} classes)")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": current_lr,
            "duration_sec": round(epoch_duration, 2),
        }
        history.append(record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | "
            f"Time: {epoch_duration:.1f}s"
        )

        if progress_callback:
            progress_callback(epoch, epochs, record)

        # Checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(
                model=model,
                architecture=arch,
                class_names=class_names,
                filepath=best_checkpoint_path,
                image_size=image_size,
                val_loss=best_val_loss,
                val_accuracy=best_val_acc,
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
            )
            print(f"  --> Checkpoint saved (val_loss: {val_loss:.4f}, val_acc: {val_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    total_training_time = time.time() - start_total_time
    logger.info(f"Training completed in {format_time(total_training_time)}")

    # Save training history CSV
    history_df = pd.DataFrame(history)
    history_csv = results_path / f"{arch}_history.csv"
    history_df.to_csv(history_csv, index=False)

    # Plot training curves
    fig_path = results_path / f"{arch}_training_curves.png"
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#007AFF", lw=2)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#FF3B30", lw=2)
    plt.title(f"{arch} - Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Acc", color="#34C759", lw=2)
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Val Acc", color="#AF52DE", lw=2)
    plt.title(f"{arch} - Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Log to SQLite Database
    exp_id = log_experiment(
        experiment_name=f"{arch}_{int(time.time())}",
        architecture=arch,
        epochs=len(history),
        batch_size=batch_size,
        learning_rate=lr,
        image_size=image_size,
        device=str(device),
        class_names=class_names,
        best_val_loss=round(best_val_loss, 4),
        best_val_accuracy=round(best_val_acc, 4),
        checkpoint_path=str(best_checkpoint_path),
        training_duration_seconds=round(total_training_time, 2),
    )

    return {
        "experiment_id": exp_id,
        "architecture": arch,
        "class_names": class_names,
        "num_classes": num_classes,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "checkpoint_path": str(best_checkpoint_path),
        "history_csv": str(history_csv),
        "curves_png": str(fig_path),
        "total_time_sec": total_training_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Train vehicle classifier models.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--arch", type=str, default="custom_cnn", choices=["custom_cnn", "resnet18", "mobilenet_v3_small"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay for regularization")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Image resolution")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class imbalance weighting")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save checkpoints")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save logs and plots")

    args = parser.parse_args()

    try:
        results = train_classifier(
            data_dir=Path(args.data_dir),
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            seed=args.seed,
            device_name=args.device,
            patience=args.patience,
            use_class_weights=not args.no_class_weights,
            save_dir=Path(args.save_dir),
            results_dir=Path(args.results_dir),
        )
        print("\n=== Training Completed Successfully ===")
        print(f"Checkpoint saved: {results['checkpoint_path']}")
        print(f"Best Val Accuracy: {results['best_val_acc']*100:.2f}% | Best Val Loss: {results['best_val_loss']:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
