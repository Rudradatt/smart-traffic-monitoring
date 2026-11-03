"""
Central configuration repository for paths, class taxonomies, training defaults,
and hardware acceleration fallback strategies across the traffic monitoring pipeline.
"""

from pathlib import Path
import os
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Guarantee operational directory structure on import so downstream writers don't trip over missing folders
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_PATH = RESULTS_DIR / "traffic_system.db"

# Core 7-class taxonomy tailored for mixed traffic conditions
VEHICLE_CLASSES = [
    "auto_rickshaw",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "scooter",
    "truck",
]

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Distinct color coding per category: BGR for OpenCV canvas drawing, Hex for web UI and analytics plots
CLASS_COLORS_BGR = {
    "auto_rickshaw": (0, 165, 255),
    "bicycle": (255, 255, 0),
    "bus": (0, 0, 255),
    "car": (0, 255, 0),
    "motorcycle": (255, 0, 128),
    "scooter": (0, 255, 255),
    "truck": (255, 128, 0),
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

# ImageNet channel statistics required when feeding transfer backbones or standard vision transforms
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 5

CLASSIFIER_ARCHITECTURES = ["custom_cnn", "resnet18", "mobilenet_v3_small"]

# COCO 80-class mapping subset. Notice standard COCO lacks 'auto_rickshaw' and aggregates all
# motorized two-wheelers under class 3 ('motorcycle'). Downstream modules use our secondary CNN
# classifier crop-refinement step to tease apart scooters from motorbikes and auto-rickshaws.
COCO_VEHICLE_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Empirical count thresholds for visual congestion buckets within a standard traffic camera FOV
DEFAULT_DENSITY_THRESHOLDS = {
    "low_max": 4,
    "medium_max": 10,
}

# Optical occlusion heuristics: cumulative vehicle bounding-box footprint relative to camera frame / ROI area
DEFAULT_OCCUPANCY_THRESHOLDS = {
    "low_max": 0.15,
    "medium_max": 0.35,
}


def get_device(requested: str = "auto") -> torch.device:
    """
    Resolve target compute device with seamless CUDA detection and graceful CPU fallback.
    Respects the DEVICE environment variable if explicitly configured.
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

