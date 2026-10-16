"""
SQLite Database Module with Versioned Schema for Experiment Tracking & Traffic Analytics.
Ensures zero data invention by recording real experiment metrics and monitoring logs.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from config import DEFAULT_DB_PATH

SCHEMA_VERSION = 1


@contextmanager
def get_db_connection(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for safe, thread-safe SQLite connection handling."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
    """Initialize database tables with schema migration checks."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Schema version table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # Check existing version
        cursor.execute("SELECT MAX(version) FROM schema_version;")
        row = cursor.fetchone()
        current_v = row[0] if (row and row[0] is not None) else 0

        if current_v < 1:
            # Table 1: Experiments
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    epochs INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL,
                    learning_rate REAL NOT NULL,
                    image_size INTEGER NOT NULL,
                    best_val_loss REAL,
                    best_val_accuracy REAL,
                    test_accuracy REAL,
                    test_macro_f1 REAL,
                    checkpoint_path TEXT,
                    metrics_json_path TEXT,
                    status TEXT NOT NULL,
                    device TEXT NOT NULL,
                    training_duration_seconds REAL,
                    class_names TEXT NOT NULL
                );
                """
            )

            # Table 2: Model Comparisons
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comparison_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    models_evaluated TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                );
                """
            )

            # Table 3: Traffic Monitoring Sessions
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS traffic_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    video_source TEXT NOT NULL,
                    total_frames INTEGER,
                    processed_frames INTEGER,
                    duration_seconds REAL,
                    fps REAL,
                    total_vehicles_counted INTEGER DEFAULT 0,
                    class_counts_json TEXT NOT NULL,
                    peak_density_level TEXT,
                    avg_density_count REAL,
                    detector_used TEXT,
                    tracker_used TEXT,
                    classifier_refinement_used INTEGER DEFAULT 0,
                    notes TEXT
                );
                """
            )

            # Table 4: Frame-level Traffic Density Logs
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS traffic_density_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    frame_index INTEGER NOT NULL,
                    timestamp_sec REAL NOT NULL,
                    vehicle_count INTEGER NOT NULL,
                    density_level TEXT NOT NULL,
                    density_ratio REAL,
                    cumulative_counts_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES traffic_sessions(id) ON DELETE CASCADE
                );
                """
            )

            cursor.execute("INSERT INTO schema_version (version) VALUES (1);")


def log_experiment(
    experiment_name: str,
    architecture: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: int,
    device: str,
    class_names: List[str],
    best_val_loss: Optional[float] = None,
    best_val_accuracy: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
    metrics_json_path: Optional[str] = None,
    status: str = "completed",
    training_duration_seconds: Optional[float] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> int:
    """Log or update an experiment run."""
    init_db(db_path)
    timestamp = datetime.now().isoformat()
    classes_json = json.dumps(class_names)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO experiments (
                experiment_name, architecture, timestamp, epochs, batch_size,
                learning_rate, image_size, best_val_loss, best_val_accuracy,
                checkpoint_path, metrics_json_path, status, device,
                training_duration_seconds, class_names
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_name,
                architecture,
                timestamp,
                epochs,
                batch_size,
                learning_rate,
                image_size,
                best_val_loss,
                best_val_accuracy,
                str(checkpoint_path) if checkpoint_path else None,
                str(metrics_json_path) if metrics_json_path else None,
                status,
                device,
                training_duration_seconds,
                classes_json,
            ),
        )
        return cursor.lastrowid


def update_experiment_test_metrics(
    experiment_id: int,
    test_accuracy: float,
    test_macro_f1: float,
    metrics_json_path: str,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    """Update experiment record with test evaluation metrics."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE experiments
            SET test_accuracy = ?, test_macro_f1 = ?, metrics_json_path = ?
            WHERE id = ?
            """,
            (test_accuracy, test_macro_f1, str(metrics_json_path), experiment_id),
        )


def get_experiments(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all experiments ordered by latest timestamp."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments ORDER BY id DESC")
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("class_names"):
                try:
                    d["class_names"] = json.loads(d["class_names"])
                except Exception:
                    pass
            result.append(d)
        return result


def get_latest_experiment(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent experiment."""
    exps = get_experiments(db_path)
    return exps[0] if exps else None


def log_model_comparison(
    comparison_name: str,
    models_evaluated: List[str],
    summary: Dict[str, Any],
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> int:
    """Record a multi-model benchmark comparison."""
    init_db(db_path)
    timestamp = datetime.now().isoformat()
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_comparisons (
                comparison_name, timestamp, models_evaluated, summary_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                comparison_name,
                timestamp,
                json.dumps(models_evaluated),
                json.dumps(summary),
            ),
        )
        return cursor.lastrowid


def get_model_comparisons(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all model benchmark comparisons."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_comparisons ORDER BY id DESC")
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("models_evaluated"):
                try:
                    d["models_evaluated"] = json.loads(d["models_evaluated"])
                except Exception:
                    pass
            if d.get("summary_json"):
                try:
                    d["summary"] = json.loads(d["summary_json"])
                except Exception:
                    pass
            result.append(d)
        return result


def log_traffic_session(
    session_name: str,
    video_source: str,
    total_frames: int,
    processed_frames: int,
    duration_seconds: float,
    fps: float,
    total_vehicles_counted: int,
    class_counts: Dict[str, int],
    peak_density_level: str,
    avg_density_count: float,
    detector_used: str,
    tracker_used: str,
    classifier_refinement_used: bool = False,
    notes: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> int:
    """Record summary of a traffic video monitoring run."""
    init_db(db_path)
    timestamp = datetime.now().isoformat()
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO traffic_sessions (
                session_name, timestamp, video_source, total_frames,
                processed_frames, duration_seconds, fps,
                total_vehicles_counted, class_counts_json, peak_density_level,
                avg_density_count, detector_used, tracker_used,
                classifier_refinement_used, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_name,
                timestamp,
                video_source,
                total_frames,
                processed_frames,
                duration_seconds,
                fps,
                total_vehicles_counted,
                json.dumps(class_counts),
                peak_density_level,
                avg_density_count,
                detector_used,
                tracker_used,
                1 if classifier_refinement_used else 0,
                notes,
            ),
        )
        return cursor.lastrowid


def log_density_records(
    session_id: int,
    records: List[Dict[str, Any]],
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    """Bulk insert per-frame density log entries."""
    if not records:
        return
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        data = [
            (
                session_id,
                r.get("frame_index", 0),
                r.get("timestamp_sec", 0.0),
                r.get("vehicle_count", 0),
                r.get("density_level", "Low"),
                r.get("density_ratio", 0.0),
                json.dumps(r.get("cumulative_counts", {})),
            )
            for r in records
        ]
        cursor.executemany(
            """
            INSERT INTO traffic_density_logs (
                session_id, frame_index, timestamp_sec, vehicle_count,
                density_level, density_ratio, cumulative_counts_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )


def get_traffic_sessions(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all traffic monitoring sessions."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM traffic_sessions ORDER BY id DESC")
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("class_counts_json"):
                try:
                    d["class_counts"] = json.loads(d["class_counts_json"])
                except Exception:
                    d["class_counts"] = {}
            result.append(d)
        return result


def get_traffic_session_logs(
    session_id: int, db_path: Union[str, Path] = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieve per-frame density records for a specific session."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM traffic_density_logs
            WHERE session_id = ?
            ORDER BY frame_index ASC
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("cumulative_counts_json"):
                try:
                    d["cumulative_counts"] = json.loads(d["cumulative_counts_json"])
                except Exception:
                    d["cumulative_counts"] = {}
            result.append(d)
        return result
