"""
Neural Network Architectures for Vehicle Classification.
Implements:
1. CustomTrafficCNN - Custom CNN built from scratch with Conv, BatchNorm, ReLU, Pooling, Dropout.
2. ResNet18Transfer - ResNet18 with transfer learning and adaptable classification head.
3. MobileNetV3Transfer - MobileNetV3-Small with lightweight transfer learning.
Dynamically infers class count from dataset class names.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
)

from config import CLASSIFIER_ARCHITECTURES, NORM_MEAN, NORM_STD


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns which channels matter for each image."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class _ResBlock(nn.Module):
    """Two 3x3 convs + skip connection + channel attention."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = _SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + identity)


class CustomTrafficCNN(nn.Module):
    """
    Custom CNN built from scratch (no pretrained weights) for vehicle classification.
    Residual blocks + channel attention + avg/max pooling head + Kaiming init.
    Same constructor/forward interface as before, so build_classifier,
    save_checkpoint and load_checkpoint work unchanged.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Stem: 224 -> 112
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stages
        self.features = nn.Sequential(
            _ResBlock(32, 64, stride=2),    # 112 -> 56
            _ResBlock(64, 64),
            _ResBlock(64, 128, stride=2),   # 56 -> 28
            _ResBlock(128, 128),
            _ResBlock(128, 256, stride=2),  # 28 -> 14
            _ResBlock(256, 256),
            _ResBlock(256, 256, stride=2),  # 14 -> 7
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Head on concatenated avg+max features (256 + 256 = 512)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.classifier(x)


class ResNet18Transfer(nn.Module):
    """
    ResNet-18 Transfer Learning baseline.
    Adapts the final fully connected layer to the dynamic number of vehicle classes.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MobileNetV3Transfer(nn.Module):
    """
    MobileNetV3-Small Transfer Learning baseline.
    Ultra-lightweight architecture optimized for edge latency and low throughput budgets.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # MobileNetV3 classifier head has [Linear, Hardswish, Dropout, Linear]
        last_channel = self.backbone.classifier[3].in_features
        self.backbone.classifier[2] = nn.Dropout(p=dropout_rate)
        self.backbone.classifier[3] = nn.Linear(last_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_classifier(
    arch: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory function to instantiate classification models.
    """
    arch = arch.lower()
    if arch not in CLASSIFIER_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{arch}'. Supported: {CLASSIFIER_ARCHITECTURES}"
        )

    if arch == "custom_cnn":
        return CustomTrafficCNN(num_classes=num_classes)
    elif arch == "resnet18":
        return ResNet18Transfer(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    elif arch == "mobilenet_v3_small":
        return MobileNetV3Transfer(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def save_checkpoint(
    model: nn.Module,
    architecture: str,
    class_names: List[str],
    filepath: Union[str, Path],
    image_size: int = 224,
    val_loss: Optional[float] = None,
    val_accuracy: Optional[float] = None,
    epoch: Optional[int] = None,
    optimizer_state: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save complete, reproducible model checkpoint with metadata and class mapping.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "architecture": architecture,
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "image_size": image_size,
        "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "timestamp": datetime.now().isoformat(),
        "optimizer_state": optimizer_state,
    }

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    filepath: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model checkpoint, re-instantiate correct architecture, and restore state dict.
    Returns:
        (model, checkpoint_metadata_dict)
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    architecture = checkpoint.get("architecture", "custom_cnn")
    class_names = checkpoint.get("class_names", [])
    num_classes = checkpoint.get("num_classes", len(class_names))

    model = build_classifier(
        arch=architecture,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint
