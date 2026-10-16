"""
Traffic Density Estimation and Virtual Counting Line Module.
Implements:
- Unique tracked vehicle counting across a configurable virtual line
- Prevention of double-counting using track ID sets
- Density estimation proxy (Instantaneous count & Bounding box occupancy ratio)
- Low, Medium, High traffic level classification with customizable thresholds
- Vehicles Per Minute (VPM) rate computation
- Transparent scientific documentation on optical proxy vs. physical road occupancy
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from config import (
    CLASS_COLORS_BGR,
    DEFAULT_DENSITY_THRESHOLDS,
    DEFAULT_OCCUPANCY_THRESHOLDS,
    VEHICLE_CLASSES,
)
from utils import setup_logger

logger = setup_logger("traffic_density")

DENSITY_DISCLAIMER = (
    "CALIBRATION NOTICE: This traffic density measurement operates as an optical proxy "
    "using 2D camera perspective projections. Precise physical road occupancy (e.g. "
    "vehicles per lane-km) and vehicle speeds require a calibrated 3D-to-2D homography matrix "
    "and real-world distance reference markers. Results here indicate visual congestion levels."
)


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Helper to determine counter-clockwise point orientation."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def line_segments_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> bool:
    """
    Check if line segment AB intersects line segment CD.
    AB: Vehicle trajectory segment (previous centroid -> current centroid)
    CD: Virtual counting line segment
    """
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def determine_crossing_direction(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> str:
    """
    Determine crossing direction relative to line CD using 2D cross product.
    A: start point, B: end point of movement
    C: line start, D: line end
    """
    # Vector CD
    cd_x = D[0] - C[0]
    cd_y = D[1] - C[1]
    # Vector AB
    ab_x = B[0] - A[0]
    ab_y = B[1] - A[1]
    # 2D cross product CD x AB
    cross = cd_x * ab_y - cd_y * ab_x
    return "inbound" if cross > 0 else "outbound"


class TrafficDensityEstimator:
    """
    Manages virtual line crossing counts and frame-level congestion estimation.
    """

    def __init__(
        self,
        counting_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        roi_polygon: Optional[List[Tuple[int, int]]] = None,
        density_thresholds: Optional[Dict[str, int]] = None,
        occupancy_thresholds: Optional[Dict[str, float]] = None,
        allow_bidirectional: bool = True,
    ):
        # Default counting line: horizontal line across center
        self.counting_line = counting_line or ((50, 360), (1230, 360))
        self.roi_polygon = roi_polygon
        self.density_thresholds = density_thresholds or DEFAULT_DENSITY_THRESHOLDS.copy()
        self.occupancy_thresholds = occupancy_thresholds or DEFAULT_OCCUPANCY_THRESHOLDS.copy()
        self.allow_bidirectional = allow_bidirectional

        # State tracking
        self.counted_tracks: Set[int] = set()
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.direction_counts: Dict[str, int] = {"inbound": 0, "outbound": 0}
        self.total_counted: int = 0
        self.active_vehicle_count: int = 0
        self.current_density_level: str = "Low"
        self.current_occupancy_ratio: float = 0.0

    def reset(self) -> None:
        """Reset all counters and state."""
        self.counted_tracks.clear()
        self.class_counts.clear()
        self.direction_counts = {"inbound": 0, "outbound": 0}
        self.total_counted = 0
        self.active_vehicle_count = 0
        self.current_density_level = "Low"
        self.current_occupancy_ratio = 0.0

    def set_counting_line(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> None:
        """Update counting line coordinates."""
        self.counting_line = (pt1, pt2)

    def set_thresholds(self, low_max: int, medium_max: int) -> None:
        """Update count-based congestion thresholds."""
        self.density_thresholds["low_max"] = low_max
        self.density_thresholds["medium_max"] = medium_max

    def classify_density(self, vehicle_count: int, occupancy_ratio: float = 0.0) -> str:
        """
        Classify density into Low, Medium, or High based on count and occupancy proxy.
        """
        low_max = self.density_thresholds.get("low_max", 4)
        med_max = self.density_thresholds.get("medium_max", 10)

        if vehicle_count <= low_max:
            return "Low"
        elif vehicle_count <= med_max:
            return "Medium"
        else:
            return "High"

    def update(
        self,
        active_tracks: List[Dict[str, Any]],
        frame_shape: Tuple[int, int, int],
        elapsed_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Process active tracks in current frame, check line crossings,
        and compute instantaneous traffic density.
        Args:
            active_tracks: List of tracked objects from VehicleTracker
            frame_shape: (height, width, channels)
            elapsed_seconds: Total elapsed video time for VPM calculation
        Returns:
            Snapshot dictionary of traffic analytics for the frame.
        """
        frame_h, frame_w = frame_shape[:2]
        c_line_start, c_line_end = self.counting_line

        newly_crossed = []
        total_vehicle_box_area = 0

        # Check for line crossings
        for trk in active_tracks:
            tid = trk["track_id"]
            curr_c = trk["centroid"]
            prev_c = trk.get("prev_centroid")
            cname = trk.get("class_name", "car")
            x1, y1, x2, y2 = trk["box"]

            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            total_vehicle_box_area += box_area

            # Only check tracks with at least 2 centroid observations
            if prev_c is not None and tid not in self.counted_tracks:
                if line_segments_intersect(prev_c, curr_c, c_line_start, c_line_end):
                    direction = determine_crossing_direction(prev_c, curr_c, c_line_start, c_line_end)
                    self.counted_tracks.add(tid)
                    self.total_counted += 1
                    self.class_counts[cname] += 1
                    self.direction_counts[direction] = self.direction_counts.get(direction, 0) + 1

                    newly_crossed.append(
                        {
                            "track_id": tid,
                            "class_name": cname,
                            "direction": direction,
                            "centroid": curr_c,
                        }
                    )

        self.active_vehicle_count = len(active_tracks)

        # Occupancy ratio proxy: vehicle box area / frame area (or ROI area)
        roi_area = frame_h * frame_w
        if self.roi_polygon and len(self.roi_polygon) >= 3:
            pts = np.array(self.roi_polygon, np.int32)
            roi_area = max(1.0, float(cv2.contourArea(pts)))

        self.current_occupancy_ratio = min(1.0, total_vehicle_box_area / max(1.0, roi_area))
        self.current_density_level = self.classify_density(
            self.active_vehicle_count, self.current_occupancy_ratio
        )

        # Vehicles Per Minute (VPM)
        vpm = 0.0
        if elapsed_seconds > 0.5:
            vpm = round((self.total_counted / elapsed_seconds) * 60.0, 1)

        return {
            "active_vehicles": self.active_vehicle_count,
            "total_counted": self.total_counted,
            "class_counts": dict(self.class_counts),
            "direction_counts": dict(self.direction_counts),
            "density_level": self.current_density_level,
            "occupancy_ratio": round(self.current_occupancy_ratio, 3),
            "vehicles_per_minute": vpm,
            "newly_crossed": newly_crossed,
            "density_disclaimer": DENSITY_DISCLAIMER,
        }

    def draw_density_overlay(
        self,
        frame: np.ndarray,
        metrics: Dict[str, Any],
        show_hud: bool = True,
    ) -> np.ndarray:
        """
        Draw virtual counting line, density status badges, and counts onto frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw counting line
        pt1, pt2 = self.counting_line
        cv2.line(annotated, pt1, pt2, (0, 255, 255), 3)  # Yellow counting line
        # Draw line endpoints
        cv2.circle(annotated, pt1, 6, (0, 200, 255), -1)
        cv2.circle(annotated, pt2, 6, (0, 200, 255), -1)

        # Counting line label
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2)
        cv2.putText(
            annotated,
            "VIRTUAL COUNTING LINE",
            (max(10, mid_x - 110), max(20, mid_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if not show_hud:
            return annotated

        # HUD Top Banner
        hud_bg = annotated.copy()
        cv2.rectangle(hud_bg, (10, 10), (w - 10, 95), (15, 20, 35), -1)
        cv2.addWeighted(hud_bg, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (w - 10, 95), (60, 70, 90), 1)

        # Density Status Color
        d_level = metrics.get("density_level", "Low")
        if d_level == "Low":
            badge_color = (0, 255, 0)      # Green
        elif d_level == "Medium":
            badge_color = (0, 215, 255)    # Orange/Yellow
        else:
            badge_color = (0, 0, 255)      # Red

        # Density Badge
        cv2.circle(annotated, (35, 40), 12, badge_color, -1)
        cv2.putText(
            annotated,
            f"DENSITY: {d_level.upper()}",
            (55, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Active & Total Counts
        active_cnt = metrics.get("active_vehicles", 0)
        tot_cnt = metrics.get("total_counted", 0)
        vpm = metrics.get("vehicles_per_minute", 0.0)

        cv2.putText(
            annotated,
            f"Active: {active_cnt} | Total Passed: {tot_cnt} | Rate: {vpm} VPM",
            (35, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 220, 240),
            1,
            cv2.LINE_AA,
        )

        # Per-Class summary on the right side of the HUD
        class_counts = metrics.get("class_counts", {})
        if class_counts:
            items = list(class_counts.items())[:4]
            cls_text = " | ".join(f"{k[:4]}: {v}" for k, v in items)
            cv2.putText(
                annotated,
                cls_text,
                (max(200, w - 420), 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 240, 200),
                1,
                cv2.LINE_AA,
            )

        return annotated
