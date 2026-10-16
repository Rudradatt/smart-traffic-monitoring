"""
Multi-Object Tracking Module for Vehicles using ByteTrack.
Maintains persistent track IDs and motion trajectories across sequential video frames.
Handles track persistence, vehicle category filtering, trajectory smoothing,
and missing/dropped track recovery.
"""

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

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

logger = setup_logger("tracking")


class VehicleTracker:
    """
    Persistent vehicle tracker leveraging Ultralytics ByteTrack integration
    with trajectory history and optional CNN crop refinement.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_type: str = "bytetrack.yaml",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        max_trajectory_length: int = 30,
        device_name: str = "auto",
        filter_vehicles_only: bool = True,
        refiner_checkpoint: Optional[Union[str, Path]] = None,
        refiner_threshold: float = 0.45,
    ):
        self.device = get_device(device_name)
        self.model_path = model_path
        self.tracker_type = tracker_type
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_trajectory_length = max_trajectory_length
        self.filter_vehicles_only = filter_vehicles_only

        logger.info(f"Initializing VehicleTracker with {model_path} and {tracker_type}...")
        self.model = YOLO(model_path)
        self.model_names = self.model.names

        # Check 7-class or COCO
        model_classes = list(self.model_names.values())
        if all(c in model_classes for c in VEHICLE_CLASSES):
            self.class_map = {idx: name for idx, name in self.model_names.items() if name in VEHICLE_CLASSES}
        else:
            self.class_map = COCO_VEHICLE_CLASS_IDS

        # Trajectory history: track_id -> deque of (cx, cy)
        self.trajectories: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.max_trajectory_length)
        )
        # Persistent class assignments for tracks (majority voting or initial)
        self.track_class_votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.track_final_classes: Dict[int, str] = {}

        # Optional CNN crop refiner
        self.refiner_checkpoint = Path(refiner_checkpoint) if refiner_checkpoint else None
        self.refiner_threshold = refiner_threshold
        self.refiner_model = None
        if self.refiner_checkpoint and self.refiner_checkpoint.is_file():
            try:
                self.refiner_model, _ = load_checkpoint(self.refiner_checkpoint, device=self.device)
                logger.info("Loaded CNN refiner for tracker refinement.")
            except Exception as e:
                logger.warning(f"Failed to load CNN refiner: {e}")

    def reset(self) -> None:
        """Clear all active tracks and trajectory history (e.g. on new video)."""
        self.trajectories.clear()
        self.track_class_votes.clear()
        self.track_final_classes.clear()
        # Reset internal tracker state in YOLO
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor.trackers = []

    def update(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process single frame, update tracking state, and return active tracked objects.
        Returns:
            List of track dicts:
            [{
                "track_id": int,
                "box": [x1, y1, x2, y2],
                "centroid": (cx, cy),
                "prev_centroid": Optional[(px, py)],
                "class_name": str,
                "confidence": float,
                "trajectory": List[(x, y)]
            }, ...]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=str(self.device),
            verbose=False,
        )

        active_tracks: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return active_tracks

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return active_tracks

        coords = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        classes = boxes_obj.cls.cpu().numpy().astype(int)

        # Track IDs might be None if tracker hasn't assigned them yet
        track_ids = (
            boxes_obj.id.cpu().numpy().astype(int)
            if boxes_obj.id is not None
            else None
        )

        for i in range(len(coords)):
            cls_id = int(classes[i])
            conf = float(confs[i])
            x1, y1, x2, y2 = coords[i]

            # Filter vehicle classes
            if self.filter_vehicles_only:
                if cls_id not in self.class_map:
                    continue
                detected_class = self.class_map[cls_id]
            else:
                detected_class = self.model_names.get(cls_id, f"class_{cls_id}")

            # Safe bounding box clip
            x1_c = max(0, int(round(x1)))
            y1_c = max(0, int(round(y1)))
            x2_c = min(w, int(round(x2)))
            y2_c = min(h, int(round(y2)))

            # If track ID is missing from model, skip or assign negative dummy
            if track_ids is None or i >= len(track_ids) or track_ids[i] is None:
                continue
            track_id = int(track_ids[i])

            # Optional CNN refinement for ambiguous classes (e.g. motorcycle -> scooter/auto_rickshaw)
            final_class = detected_class
            if (
                self.refiner_model is not None
                and detected_class in ("motorcycle", "car")
                and (x2_c - x1_c >= 20)
                and (y2_c - y1_c >= 20)
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
                except Exception:
                    pass

            # Class vote smoothing to prevent flickering between frames
            self.track_class_votes[track_id][final_class] += 1
            stable_class = max(
                self.track_class_votes[track_id].items(), key=lambda x: x[1]
            )[0]
            self.track_final_classes[track_id] = stable_class

            # Compute centroid
            cx = int((x1_c + x2_c) / 2)
            cy = int((y1_c + y2_c) / 2)
            centroid = (cx, cy)

            prev_centroid = (
                self.trajectories[track_id][-1]
                if len(self.trajectories[track_id]) > 0
                else None
            )
            self.trajectories[track_id].append(centroid)

            active_tracks.append(
                {
                    "track_id": track_id,
                    "box": [x1_c, y1_c, x2_c, y2_c],
                    "centroid": centroid,
                    "prev_centroid": prev_centroid,
                    "class_name": stable_class,
                    "confidence": round(conf, 4),
                    "trajectory": list(self.trajectories[track_id]),
                }
            )

        return active_tracks

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        draw_trails: bool = True,
    ) -> np.ndarray:
        """
        Annotate frame with bounding boxes, persistent track IDs, and movement trails.
        """
        annotated = frame.copy()

        for trk in tracks:
            x1, y1, x2, y2 = trk["box"]
            tid = trk["track_id"]
            cname = trk["class_name"]
            conf = trk["confidence"]
            trajectory = trk["trajectory"]

            color = CLASS_COLORS_BGR.get(cname, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw trajectory trail
            if draw_trails and len(trajectory) > 1:
                pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2)

            # Centroid point
            cx, cy = trk["centroid"]
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

            # Label banner
            label = f"#{tid} {cname} {conf*100:.0f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - lh - 8)),
                (x1 + lw + 6, max(lh + 8, y1)),
                color,
                -1,
            )
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
