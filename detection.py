"""
Vehicle Detection Module using Ultralytics YOLO with Optional CNN Crop Refinement.
Supports:
- Pretrained general-purpose models (e.g., yolov8n.pt, yolov8s.pt)
- Custom-trained 7-class YOLO models
- Transparent COCO label mapping (highlighting COCO limitations for scooters/rickshaws)
- Optional CNN classifier crop refinement to distinguish sub-types (e.g. motorcycle vs scooter vs auto_rickshaw)
- Custom YOLO training YAML generation
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from classifier_models import load_checkpoint
from config import (
    CLASS_COLORS_BGR,
    COCO_VEHICLE_CLASS_IDS,
    VEHICLE_CLASSES,
    get_device,
)
from predict_image import predict_vehicle_image
from utils import setup_logger

logger = setup_logger("detection")

COCO_LIMITATION_EXPLANATION = (
    "COCO DETECTOR LIMITATION: Standard YOLO models trained on COCO detect 'bicycle', "
    "'car', 'motorcycle', 'bus', and 'truck'. They DO NOT have an 'auto_rickshaw' class "
    "and classify motorized two-wheelers (scooters, mopeds, superbikes) collectively as "
    "'motorcycle'. To achieve true 7-class detection, either enable the CNN Crop Refiner "
    "or train a custom YOLO model using the generated dataset YAML."
)


def generate_yolo_dataset_yaml(
    data_root: Union[str, Path],
    output_yaml: Union[str, Path] = "results/traffic_yolo_data.yaml",
) -> Path:
    """
    Generate dataset YAML configuration required to train custom YOLO models on 7 classes.
    """
    root_path = Path(data_root).resolve()
    yaml_path = Path(output_yaml).resolve()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""# Ultralytics YOLO 7-Class Vehicle Dataset Configuration
path: {root_path.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: auto_rickshaw
  1: bicycle
  2: bus
  3: car
  4: motorcycle
  5: scooter
  6: truck
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    logger.info(f"YOLO training configuration generated at: {yaml_path}")
    return yaml_path


class VehicleDetector:
    """
    Modular vehicle detector leveraging Ultralytics YOLO with optional CNN crop refinement.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device_name: str = "auto",
        filter_vehicles_only: bool = True,
        refiner_checkpoint: Optional[Union[str, Path]] = None,
        refiner_threshold: float = 0.45,
        refine_classes: Optional[List[str]] = None,
    ):
        self.device = get_device(device_name)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.filter_vehicles_only = filter_vehicles_only
        self.refiner_checkpoint = Path(refiner_checkpoint) if refiner_checkpoint else None
        self.refiner_threshold = refiner_threshold
        self.refine_classes = refine_classes or ["motorcycle", "car"]

        logger.info(f"Loading YOLO detector from '{model_path}' on {self.device}...")
        self.yolo_model = YOLO(model_path)
        self.model_names = self.yolo_model.names

        # Determine whether model is custom 7-class or standard COCO
        model_class_list = list(self.model_names.values())
        if all(c in model_class_list for c in VEHICLE_CLASSES):
            self.is_custom_7class = True
            self.class_map = {idx: name for idx, name in self.model_names.items() if name in VEHICLE_CLASSES}
            logger.info("Detected custom 7-class YOLO model.")
        else:
            self.is_custom_7class = False
            self.class_map = COCO_VEHICLE_CLASS_IDS
            logger.info(f"Using pretrained detector. {COCO_LIMITATION_EXPLANATION}")

        # Optional CNN crop refiner
        self.refiner_model = None
        self.refiner_meta = None
        if self.refiner_checkpoint and self.refiner_checkpoint.is_file():
            try:
                self.refiner_model, self.refiner_meta = load_checkpoint(
                    self.refiner_checkpoint, device=self.device
                )
                logger.info(f"Loaded CNN Crop Refiner from {self.refiner_checkpoint.name}")
            except Exception as e:
                logger.warning(f"Could not load CNN refiner: {e}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run object detection on an image/frame.
        Args:
            frame: BGR numpy array from OpenCV
        Returns:
            List of detection dicts:
            [{
                "box": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str,
                "refined": bool,
                "refiner_confidence": Optional[float]
            }, ...]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        results = self.yolo_model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=str(self.device),
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return detections

        coords = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        classes = boxes_obj.cls.cpu().numpy().astype(int)

        for i in range(len(coords)):
            cls_id = int(classes[i])
            conf = float(confs[i])
            x1, y1, x2, y2 = coords[i]

            # Filter vehicle classes
            if self.filter_vehicles_only:
                if cls_id not in self.class_map:
                    continue
                class_name = self.class_map[cls_id]
            else:
                class_name = self.model_names.get(cls_id, f"class_{cls_id}")

            x1_c = max(0, int(round(x1)))
            y1_c = max(0, int(round(y1)))
            x2_c = min(w, int(round(x2)))
            y2_c = min(h, int(round(y2)))

            refined = False
            refiner_conf = None
            final_class = class_name

            # Optional CNN Refinement for sub-class disambiguation
            if (
                self.refiner_model is not None
                and class_name in self.refine_classes
                and (x2_c - x1_c >= 16)
                and (y2_c - y1_c >= 16)
            ):
                crop_bgr = frame[y1_c:y2_c, x1_c:x2_c]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                try:
                    ref_res = predict_vehicle_image(
                        image_input=crop_rgb,
                        checkpoint_path=self.refiner_checkpoint,
                        confidence_threshold=self.refiner_threshold,
                        device_name=str(self.device),
                    )
                    if not ref_res["is_uncertain"]:
                        final_class = ref_res["predicted_class"]
                        refiner_conf = ref_res["confidence"]
                        refined = True
                except Exception as e:
                    logger.debug(f"CNN crop refinement error: {e}")

            detections.append(
                {
                    "box": [x1_c, y1_c, x2_c, y2_c],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": final_class,
                    "original_class": class_name,
                    "refined": refined,
                    "refiner_confidence": refiner_conf,
                }
            )

        return detections

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_confidence: bool = True,
        show_refined_badge: bool = True,
    ) -> np.ndarray:
        """
        Render bounding boxes and label tags onto frame copy.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cname = det["class_name"]
            conf = det["confidence"]
            refined = det.get("refined", False)

            color = CLASS_COLORS_BGR.get(cname, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label text
            label = f"{cname}"
            if show_confidence:
                label += f" {conf*100:.0f}%"
            if refined and show_refined_badge:
                label += " [Refined]"

            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - lh - 8)), (x1 + lw + 6, max(lh + 8, y1)), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(lh + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated
