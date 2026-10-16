"""
Traffic Analytics and Statistical Reporting Module.
Processes frame logs, computes congestion trends, aggregates vehicle distributions,
generates analytic charts, and handles CSV/JSON export.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLASS_COLORS_HEX, RESULTS_DIR
from database import (
    get_traffic_session_logs,
    get_traffic_sessions,
    log_density_records,
    log_traffic_session,
)
from utils import save_json, setup_logger

logger = setup_logger("analytics")


def summarize_traffic_session(
    frame_logs: List[Dict[str, Any]],
    session_name: str,
    video_source: str,
    fps: float,
    detector_used: str,
    tracker_used: str,
    classifier_refinement_used: bool = False,
    notes: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Summarize frame-by-frame monitoring logs and record the session into SQLite.
    Returns:
        (session_id, summary_dict)
    """
    if not frame_logs:
        return 0, {}

    total_frames = len(frame_logs)
    duration_sec = total_frames / max(fps, 1.0)

    # Final cumulative counts from the last logged frame
    last_log = frame_logs[-1]
    last_cum_counts = last_log.get("cumulative_counts", {})
    total_vehicles_counted = sum(last_cum_counts.values()) if last_cum_counts else 0

    # Density breakdown
    counts = [log.get("vehicle_count", 0) for log in frame_logs]
    density_levels = [log.get("density_level", "Low") for log in frame_logs]

    avg_density_count = round(float(np.mean(counts)), 2) if counts else 0.0
    peak_count = max(counts) if counts else 0

    # Peak density level
    level_counts = Counter(density_levels)
    most_common_level = level_counts.most_common(1)[0][0] if level_counts else "Low"
    peak_level = (
        "High" if "High" in density_levels
        else ("Medium" if "Medium" in density_levels else "Low")
    )

    session_id = log_traffic_session(
        session_name=session_name,
        video_source=video_source,
        total_frames=total_frames,
        processed_frames=total_frames,
        duration_seconds=round(duration_sec, 2),
        fps=round(fps, 2),
        total_vehicles_counted=total_vehicles_counted,
        class_counts=last_cum_counts,
        peak_density_level=peak_level,
        avg_density_count=avg_density_count,
        detector_used=detector_used,
        tracker_used=tracker_used,
        classifier_refinement_used=classifier_refinement_used,
        notes=notes,
    )

    # Bulk log frame records
    log_density_records(session_id, frame_logs)

    summary = {
        "session_id": session_id,
        "session_name": session_name,
        "total_frames": total_frames,
        "duration_seconds": round(duration_sec, 2),
        "fps": round(fps, 2),
        "total_vehicles_counted": total_vehicles_counted,
        "class_counts": last_cum_counts,
        "avg_vehicles_per_frame": avg_density_count,
        "peak_vehicles_in_frame": peak_count,
        "peak_density_level": peak_level,
        "predominant_density_level": most_common_level,
        "density_distribution": dict(level_counts),
    }

    return session_id, summary


def generate_session_plots(
    frame_logs: List[Dict[str, Any]],
    output_dir: Union[str, Path] = RESULTS_DIR,
    session_id: Union[int, str] = "latest",
) -> Dict[str, str]:
    """
    Generate and save analytics figures (Density Timeline & Class Distribution).
    Returns paths to generated image files.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not frame_logs:
        return {}

    df = pd.DataFrame(frame_logs)

    # 1. Density and Vehicle Count Timeline
    timeline_path = out_path / f"session_{session_id}_density_timeline.png"
    plt.figure(figsize=(10, 4.5))
    if "timestamp_sec" in df.columns:
        x = df["timestamp_sec"]
        plt.xlabel("Time (seconds)")
    else:
        x = df["frame_index"]
        plt.xlabel("Frame Index")

    plt.plot(x, df["vehicle_count"], color="#007AFF", lw=2, label="Active Vehicles")
    plt.axhline(y=4, color="#34C759", linestyle="--", alpha=0.7, label="Low Threshold (<=4)")
    plt.axhline(y=10, color="#FF9500", linestyle="--", alpha=0.7, label="Medium Threshold (<=10)")

    plt.title("Traffic Density Profile Over Time", fontsize=12)
    plt.ylabel("Vehicle Count in Scene")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(timeline_path, dpi=150)
    plt.close()

    # 2. Cumulative Class Distribution Donut Chart
    dist_path = out_path / f"session_{session_id}_class_distribution.png"
    last_cum = frame_logs[-1].get("cumulative_counts", {})
    if last_cum and sum(last_cum.values()) > 0:
        labels = list(last_cum.keys())
        values = list(last_cum.values())
        colors = [CLASS_COLORS_HEX.get(k, "#8E8E93") for k in labels]

        plt.figure(figsize=(6, 6))
        plt.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.4, edgecolor="w"),
        )
        plt.title("Tracked Vehicle Class Distribution", fontsize=12)
        plt.tight_layout()
        plt.savefig(dist_path, dpi=150)
        plt.close()
    else:
        dist_path = ""

    return {
        "density_timeline": str(timeline_path),
        "class_distribution": str(dist_path) if dist_path else None,
    }


def export_session_reports(
    session_id: int,
    output_dir: Union[str, Path] = RESULTS_DIR,
) -> Dict[str, str]:
    """
    Export full CSV and JSON reports for a stored session.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    sessions = get_traffic_sessions()
    session = next((s for s in sessions if s["id"] == session_id), None)
    if not session:
        raise ValueError(f"Session ID {session_id} not found in database.")

    logs = get_traffic_session_logs(session_id)

    # Export frame logs to CSV
    csv_path = out_path / f"traffic_session_{session_id}_logs.csv"
    if logs:
        df = pd.DataFrame(logs)
        df.to_csv(csv_path, index=False)

    # Export session metadata & summary to JSON
    json_path = out_path / f"traffic_session_{session_id}_summary.json"
    full_data = {
        "session_metadata": session,
        "total_records": len(logs),
    }
    save_json(full_data, json_path)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
