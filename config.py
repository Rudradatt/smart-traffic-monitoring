"""
Global Configuration for Smart Traffic Monitoring and Vehicle Classification System.
Provides paths, vehicle classes, model parameters, and hardware settings.
"""

from pathlib import Path
import os
import torch

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure essential directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database Path
DEFAULT_DB_PATH = RESULTS_DIR / "traffic_system.db"

# Canonical 7 Vehicle Classes
VEHICLE_CLASSES = [
    "auto_rickshaw",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "scooter",
    "truck",
]

# Supported Image Extensions
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Color palette for 7 classes (BGR for OpenCV, Hex for Streamlit/Matplotlib)
CLASS_COLORS_BGR = {
    "auto_rickshaw": (0, 165, 255),    # Orange
    "bicycle": (255, 255, 0),          # Cyan
    "bus": (0, 0, 255),                # Red
    "car": (0, 255, 0),                # Green
    "motorcycle": (255, 0, 128),       # Purple
    "scooter": (0, 255, 255),          # Yellow
    "truck": (255, 128, 0),            # Blue-ish
}

CLASS_COLORS_HEX = {
    "auto_rickshaw": "#FFA500",
    "bicycle": "#00FFFF",
    "bus": "#FF3B30",
    "car": "#34C759",
    "motorcycle": "#AF52DE",
    "scooter": "#FFCC00",
    "truck": "#007AFF",
}

# Normalization constants (standard ImageNet)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Default Training Parameters
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 5

# Supported Classification Architectures
CLASSIFIER_ARCHITECTURES = ["custom_cnn", "resnet18", "mobilenet_v3_small"]

# COCO Vehicle Classes Mapping for standard YOLOv8 models
# Notice: COCO standard dataset has no auto_rickshaw, and lumps scooters into motorcycle.
COCO_VEHICLE_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Traffic Density Proxy Thresholds
# Count-based thresholds (instantaneous vehicles in frame / ROI)
DEFAULT_DENSITY_THRESHOLDS = {
    "low_max": 4,      # 0 - 4 vehicles: Low
    "medium_max": 10,  # 5 - 10 vehicles: Medium
    # > 10 vehicles: High
}

# Occupancy ratio thresholds (sum of vehicle bounding box areas / ROI area)
DEFAULT_OCCUPANCY_THRESHOLDS = {
    "low_max": 0.15,     # < 15% occupancy: Low
    "medium_max": 0.35,  # 15% - 35% occupancy: Medium
    # > 35% occupancy: High
}


def get_device(requested: str = "auto") -> torch.device:
    """
    Get computation device with automatic CUDA detection and CPU fallback.
    Args:
        requested: 'auto', 'cuda', or 'cpu'
    Returns:
        torch.device
    """
    req = os.getenv("DEVICE", requested).lower()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARNING] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    elif req == "cpu":
        return torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
