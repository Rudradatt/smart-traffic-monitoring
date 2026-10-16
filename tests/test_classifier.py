"""
Unit tests for vehicle classifier architectures and checkpoint handlers.
Ensures zero external network dependencies during testing (pretrained=False).
"""

from pathlib import Path
import pytest
import torch

from classifier_models import (
    CustomTrafficCNN,
    MobileNetV3Transfer,
    ResNet18Transfer,
    build_classifier,
    load_checkpoint,
    save_checkpoint,
)
from utils import calculate_model_parameters


def test_custom_cnn_forward_7classes():
    model = CustomTrafficCNN(num_classes=7)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)


def test_custom_cnn_dynamic_num_classes():
    model = CustomTrafficCNN(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 4)


def test_resnet18_transfer_forward_no_weights():
    model = ResNet18Transfer(num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)


def test_mobilenet_v3_transfer_forward_no_weights():
    model = MobileNetV3Transfer(num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)


def test_build_classifier_factory():
    m1 = build_classifier("custom_cnn", num_classes=5, pretrained=False)
    assert isinstance(m1, CustomTrafficCNN)

    m2 = build_classifier("resnet18", num_classes=5, pretrained=False)
    assert isinstance(m2, ResNet18Transfer)

    m3 = build_classifier("mobilenet_v3_small", num_classes=5, pretrained=False)
    assert isinstance(m3, MobileNetV3Transfer)

    with pytest.raises(ValueError):
        build_classifier("non_existent_arch", num_classes=5)


def test_calculate_model_parameters():
    model = CustomTrafficCNN(num_classes=7)
    params = calculate_model_parameters(model)
    assert "total_params" in params
    assert "trainable_params" in params
    assert params["total_params"] > 0
    assert params["trainable_params"] == params["total_params"]


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    cp_path = tmp_path / "test_model.pth"
    class_names = ["auto_rickshaw", "car", "motorcycle"]
    model = CustomTrafficCNN(num_classes=len(class_names))

    save_checkpoint(
        model=model,
        architecture="custom_cnn",
        class_names=class_names,
        filepath=cp_path,
        image_size=224,
        val_loss=0.45,
        val_accuracy=0.88,
        epoch=5,
    )

    assert cp_path.is_file()

    loaded_model, meta = load_checkpoint(cp_path, device=torch.device("cpu"))
    assert meta["architecture"] == "custom_cnn"
    assert meta["class_names"] == class_names
    assert meta["val_accuracy"] == 0.88
    assert meta["epoch"] == 5

    # Test forward pass with loaded model
    dummy = torch.randn(1, 3, 224, 224)
    out = loaded_model(dummy)
    assert out.shape == (1, len(class_names))
