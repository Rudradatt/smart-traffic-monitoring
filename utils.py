"""
General Utility Functions for Smart Traffic Monitoring System.
Includes seed initialization, device inspection, file I/O, and model inspection.
"""

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across Python, NumPy, and PyTorch for reproducible runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, Any]:
    """
    Return detailed hardware and device information.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": "cuda" if torch.cuda.is_available() else "cpu",
        "pytorch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        try:
            device_idx = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(device_idx)
            props = torch.cuda.get_device_properties(device_idx)
            info["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
            info["cuda_capability"] = f"{props.major}.{props.minor}"
        except Exception as e:
            info["cuda_info_error"] = str(e)
    else:
        info["device_name"] = "CPU"
        info["total_vram_gb"] = 0.0

    return info


def format_time(seconds: float) -> str:
    """Format duration in seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save dictionary to formatted JSON file with parent folder creation."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    def serializer(obj):
        if isinstance(obj, (datetime, np.datetime64)):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=serializer)


def load_json(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Safely load JSON file or return None if missing or corrupt."""
    path = Path(filepath)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger("utils").error(f"Failed to read {filepath}: {e}")
        return None


def calculate_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculate total and trainable parameters for any PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
    }


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """Return file size in megabytes."""
    path = Path(filepath)
    if not path.is_file():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 2)


def setup_logger(name: str = "traffic_system", level: int = logging.INFO) -> logging.Logger:
    """Configure a clean standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
