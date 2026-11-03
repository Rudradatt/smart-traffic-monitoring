"""
Unit tests for traffic density estimation and virtual counting line algorithms.
"""

import pytest
from traffic_density import (
    TrafficDensityEstimator,
    determine_crossing_direction,
    line_segments_intersect,
)


def test_line_segments_intersect_true():
    line_start = (0, 100)
    line_end = (200, 100)

    traj_start = (100, 50)
    traj_end = (100, 150)

    assert line_segments_intersect(traj_start, traj_end, line_start, line_end) is True


def test_line_segments_intersect_false_parallel():
    line_start = (0, 100)
    line_end = (200, 100)

    # Parallel trajectory that never intersects the cordon line
    traj_start = (0, 50)
    traj_end = (200, 50)

    assert line_segments_intersect(traj_start, traj_end, line_start, line_end) is False


def test_line_segments_intersect_false_short():
    line_start = (0, 100)
    line_end = (200, 100)

    # Trajectory moving towards the line that terminates prior to crossing
    traj_start = (100, 20)
    traj_end = (100, 80)

    assert line_segments_intersect(traj_start, traj_end, line_start, line_end) is False


def test_determine_crossing_direction():
    line_start = (0, 100)
    line_end = (200, 100)

    dir_down = determine_crossing_direction((100, 50), (100, 150), line_start, line_end)
    dir_up = determine_crossing_direction((100, 150), (100, 50), line_start, line_end)

    assert dir_down != dir_up


def test_unique_counting_and_no_duplicate_counts():
    estimator = TrafficDensityEstimator(counting_line=((0, 100), (200, 100)))

    # Frame 1: Initial detection above the line; single-point centroid cannot trigger crossing
    frame1_tracks = [
        {"track_id": 1, "centroid": (100, 80), "prev_centroid": None, "box": [80, 60, 120, 100], "class_name": "car"}
    ]
    res1 = estimator.update(frame1_tracks, frame_shape=(400, 400, 3), elapsed_seconds=1.0)
    assert res1["total_counted"] == 0

    # Frame 2: Centroid transitions across the line segment
    frame2_tracks = [
        {"track_id": 1, "centroid": (100, 120), "prev_centroid": (100, 80), "box": [80, 100, 120, 140], "class_name": "car"}
    ]
    res2 = estimator.update(frame2_tracks, frame_shape=(400, 400, 3), elapsed_seconds=2.0)
    assert res2["total_counted"] == 1
    assert res2["class_counts"]["car"] == 1

    # Frame 3: Continued motion on the far side of the line must not re-trigger counting
    frame3_tracks = [
        {"track_id": 1, "centroid": (100, 160), "prev_centroid": (100, 120), "box": [80, 140, 120, 180], "class_name": "car"}
    ]
    res3 = estimator.update(frame3_tracks, frame_shape=(400, 400, 3), elapsed_seconds=3.0)
    assert res3["total_counted"] == 1


def test_density_classification_thresholds():
    estimator = TrafficDensityEstimator(density_thresholds={"low_max": 3, "medium_max": 8})

    assert estimator.classify_density(2) == "Low"
    assert estimator.classify_density(3) == "Low"
    assert estimator.classify_density(4) == "Medium"
    assert estimator.classify_density(8) == "Medium"
    assert estimator.classify_density(9) == "High"
    assert estimator.classify_density(20) == "High"


def test_vpm_calculation():
    estimator = TrafficDensityEstimator(counting_line=((0, 100), (200, 100)))

    for tid in range(1, 6):
        estimator.update(
            [{"track_id": tid, "centroid": (100, 120), "prev_centroid": (100, 80), "box": [80, 100, 120, 140], "class_name": "car"}],
            frame_shape=(400, 400, 3),
            elapsed_seconds=30.0,
        )

    res = estimator.update([], frame_shape=(400, 400, 3), elapsed_seconds=30.0)
    assert res["total_counted"] == 5
    # 5 vehicles crossing in a 30-second window scales linearly to 10 vehicles per minute
    assert res["vehicles_per_minute"] == 10.0
