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

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
)

from config import CLASSIFIER_ARCHITECTURES, NORM_MEAN, NORM_STD


# =====================================================================
# CUSTOM CNN - every layer is written by hand in NumPy (forward AND backward).
# No nn.Conv2d / nn.BatchNorm2d / nn.ReLU / nn.MaxPool2d / nn.Linear / nn.Dropout
# is used. The maths lives in the _Np* classes below.
#
# CustomTrafficCNN (further down) is only a thin shell that lets the rest of your
# project (build_classifier, train_classifier.py, save_checkpoint, load_checkpoint)
# keep working unchanged: it stores the weights as torch Parameters so your
# existing AdamW optimizer can update them, and it hands the real computation
# to the hand-written NumPy layers through one custom autograd function.
# =====================================================================
class _P:
    """A trainable NumPy tensor plus its gradient."""

    def __init__(self, value):
        self.value = value.astype(np.float32)
        self.grad = np.zeros_like(self.value)


class _NpConv2D:
    """kxk convolution using im2col: gather every patch, then ONE matrix multiply."""

    def __init__(self, in_ch, out_ch, rng, k=3, stride=1, pad=1):
        fan_in = in_ch * k * k
        self.W = _P(rng.standard_normal((out_ch, in_ch, k, k)) * np.sqrt(2.0 / fan_in))
        self.b = _P(np.zeros(out_ch))
        self.k, self.s, self.p = k, stride, pad

    def params(self):
        return [self.W, self.b]

    def forward(self, x, training):
        N, C, H, W = x.shape
        k, s, p = self.k, self.s, self.p
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x
        win = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::s, ::s]  # N,C,oh,ow,k,k
        oh, ow = win.shape[2], win.shape[3]
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * oh * ow, C * k * k)
        F = self.W.value.shape[0]
        out = cols @ self.W.value.reshape(F, -1).T + self.b.value
        self.cache = (cols, x.shape, oh, ow)
        return np.ascontiguousarray(out.reshape(N, oh, ow, F).transpose(0, 3, 1, 2))

    def backward(self, dout):
        cols, (N, C, H, W), oh, ow = self.cache
        k, s, p = self.k, self.s, self.p
        F = self.W.value.shape[0]
        d2 = dout.transpose(0, 2, 3, 1).reshape(-1, F)
        self.W.grad += (d2.T @ cols).reshape(self.W.value.shape)
        self.b.grad += d2.sum(axis=0)
        dcols = (d2 @ self.W.value.reshape(F, -1)).reshape(N, oh, ow, C, k, k)
        dxp = np.zeros((N, C, H + 2 * p, W + 2 * p), dtype=dout.dtype)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + s * oh:s, j:j + s * ow:s] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, p:p + H, p:p + W] if p else dxp


class _NpBatchNorm2D:
    def __init__(self, ch, momentum=0.1, eps=1e-5):
        self.gamma = _P(np.ones(ch))
        self.beta = _P(np.zeros(ch))
        self.running_mean = np.zeros(ch, np.float32)
        self.running_var = np.ones(ch, np.float32)
        self.momentum, self.eps = momentum, eps

    def params(self):
        return [self.gamma, self.beta]

    def forward(self, x, training):
        ax = (0, 2, 3)
        if training:
            mean, var = x.mean(axis=ax), x.var(axis=ax)
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mean
            self.running_var = (1 - m) * self.running_var + m * var
        else:
            mean, var = self.running_mean, self.running_var
        invstd = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mean[None, :, None, None]) * invstd[None, :, None, None]
        self.cache = (xhat, invstd)
        return self.gamma.value[None, :, None, None] * xhat + self.beta.value[None, :, None, None]

    def backward(self, dout):
        xhat, invstd = self.cache
        ax = (0, 2, 3)
        m = dout.shape[0] * dout.shape[2] * dout.shape[3]
        self.beta.grad += dout.sum(axis=ax)
        self.gamma.grad += (dout * xhat).sum(axis=ax)
        dxhat = dout * self.gamma.value[None, :, None, None]
        s1 = dxhat.sum(axis=ax, keepdims=True)
        s2 = (dxhat * xhat).sum(axis=ax, keepdims=True)
        return (invstd[None, :, None, None] / m) * (m * dxhat - s1 - xhat * s2)


class _NpReLU:
    def params(self):
        return []

    def forward(self, x, training):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class _NpMaxPool2:
    """2x2 max pooling, stride 2."""

    def params(self):
        return []

    def forward(self, x, training):
        N, C, H, W = x.shape
        H2, W2 = H // 2, W // 2
        xr = x[:, :, :H2 * 2, :W2 * 2].reshape(N, C, H2, 2, W2, 2)
        out = xr.max(axis=(3, 5))
        self.mask = xr == out[:, :, :, None, :, None]
        self.shape = x.shape
        return out

    def backward(self, dout):
        N, C, H, W = self.shape
        H2, W2 = H // 2, W // 2
        share = (self.mask / self.mask.sum(axis=(3, 5), keepdims=True)).astype(dout.dtype)
        dxr = share * dout[:, :, :, None, :, None]
        dx = np.zeros(self.shape, dtype=dout.dtype)
        dx[:, :, :H2 * 2, :W2 * 2] = dxr.reshape(N, C, H2 * 2, W2 * 2)
        return dx


class _NpGlobalAvgPool:
    def params(self):
        return []

    def forward(self, x, training):
        self.shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, dout):
        N, C, H, W = self.shape
        return np.broadcast_to((dout / (H * W))[:, :, None, None], self.shape).copy()


class _NpDropout:
    def __init__(self, rate, rng):
        self.rate, self.rng = rate, rng

    def params(self):
        return []

    def forward(self, x, training):
        if not training or self.rate == 0:
            self.mask = None
            return x
        keep = 1.0 - self.rate
        self.mask = (self.rng.random(x.shape) < keep).astype(x.dtype) / keep
        return x * self.mask

    def backward(self, dout):
        return dout if self.mask is None else dout * self.mask


class _NpDense:
    def __init__(self, n_in, n_out, rng):
        self.W = _P(rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in))
        self.b = _P(np.zeros(n_out))

    def params(self):
        return [self.W, self.b]

    def forward(self, x, training):
        self.x = x
        return x @ self.W.value + self.b.value

    def backward(self, dout):
        self.W.grad += self.x.T @ dout
        self.b.grad += dout.sum(axis=0)
        return dout @ self.W.value.T


class _NpCore:
    """
    [Conv-BN-ReLU-MaxPool] x3 -> [Conv-BN-ReLU] -> GlobalAvgPool -> Dropout -> Dense
    Works for any input size (global pooling), e.g. 64x64 -> 16x32x32 -> 32x16x16
    -> 64x8x8 -> 128x8x8 -> 128 -> num_classes.
    """

    def __init__(self, num_classes, in_channels, widths, dropout, rng):
        layers, ch = [], in_channels
        for i, w in enumerate(widths):
            layers += [_NpConv2D(ch, w, rng), _NpBatchNorm2D(w), _NpReLU()]
            if i < len(widths) - 1:
                layers.append(_NpMaxPool2())
            ch = w
        layers += [_NpGlobalAvgPool(), _NpDropout(dropout, rng), _NpDense(ch, num_classes, rng)]
        self.layers = layers

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def bn_layers(self):
        return [l for l in self.layers if isinstance(l, _NpBatchNorm2D)]

    def forward(self, x, training):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)


class _NumpyCNNFunction(torch.autograd.Function):
    """
    Bridge between torch's training loop and the hand-written NumPy layers.
    forward : torch tensor -> NumPy -> hand-written forward pass -> torch tensor
    backward: runs the hand-written backward pass and returns the weight gradients
              so the unchanged optimizer in train_classifier.py can update them.
    """

    @staticmethod
    def forward(ctx, x, core, training, *weights):
        for p, w in zip(core.params(), weights):
            p.value = w.detach().cpu().numpy()
            p.grad = np.zeros_like(p.value)
        out = core.forward(x.detach().cpu().numpy(), training)
        ctx.core, ctx.device, ctx.dtype = core, x.device, x.dtype
        return torch.from_numpy(np.ascontiguousarray(out)).to(device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        ctx.core.backward(grad_out.detach().cpu().numpy())
        grads = [torch.from_numpy(np.ascontiguousarray(p.grad)).to(ctx.device, ctx.dtype)
                 for p in ctx.core.params()]
        return (None, None, None, *grads)


class CustomTrafficCNN(nn.Module):
    """
    Custom CNN built completely from scratch: convolution, batch-norm, ReLU,
    max-pooling, global average pooling, dropout, dense layer, and all their
    backward passes are hand-written NumPy (see the _Np* classes above).
    Same constructor/forward interface as before, so build_classifier,
    save_checkpoint, load_checkpoint and train_classifier.py work unchanged.

    Tips: train with  --image-size 64 --device cpu  (NumPy runs on the CPU only).
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        self._core = _NpCore(num_classes, in_channels, (16, 32, 64, 128), dropout_rate, rng)

        # Weights live in torch Parameters (only as storage, so the optimizer and
        # state_dict can see them). Batch-norm running stats live in buffers.
        self._weight_names = []
        for i, p in enumerate(self._core.params()):
            name = f"w{i}"
            self.register_parameter(name, nn.Parameter(torch.from_numpy(p.value.copy())))
            self._weight_names.append(name)
        for i, bn in enumerate(self._core.bn_layers()):
            self.register_buffer(f"bn{i}_mean", torch.from_numpy(bn.running_mean.copy()))
            self.register_buffer(f"bn{i}_var", torch.from_numpy(bn.running_var.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bns = self._core.bn_layers()
        for i, bn in enumerate(bns):
            bn.running_mean = getattr(self, f"bn{i}_mean").detach().cpu().numpy().copy()
            bn.running_var = getattr(self, f"bn{i}_var").detach().cpu().numpy().copy()

        weights = [getattr(self, n) for n in self._weight_names]
        out = _NumpyCNNFunction.apply(x, self._core, self.training, *weights)

        if self.training:  # keep batch-norm running stats in the checkpoint
            with torch.no_grad():
                for i, bn in enumerate(bns):
                    getattr(self, f"bn{i}_mean").copy_(torch.from_numpy(np.asarray(bn.running_mean)))
                    getattr(self, f"bn{i}_var").copy_(torch.from_numpy(np.asarray(bn.running_var)))
        return out


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