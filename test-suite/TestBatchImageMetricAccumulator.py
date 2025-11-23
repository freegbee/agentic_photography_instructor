# ensure project src is on sys.path for imports when running tests from repo root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import time
import math

from experiments.shared.BatchImageMetricAccumulator import BatchImageMetricAccumulator


def test_empty_accumulator_compute_metrics():
    acc = BatchImageMetricAccumulator()
    acc.reset()
    # without start/stop should yield None duration and zero images
    metrics = acc.compute_metrics()
    assert metrics["number_of_images"] == 0
    assert metrics["images_created_count"] == 0
    assert metrics["average_initial_score"] is None
    assert metrics["average_final_score"] is None
    assert metrics["average_change"] is None
    assert metrics["total_batch_duration_seconds"] is None
    assert metrics["avg_image_duration_seconds"] is None


def test_add_images_and_compute_stats():
    acc = BatchImageMetricAccumulator()
    acc.reset()
    acc.start(step=1)
    # add three images: one with both scores, one with only final, one with only initial
    acc.add_image(1.0, 2.0)      # change = 1.0
    acc.add_image(None, 3.0)     # initial missing
    acc.add_image(0.5, None)     # final missing
    time.sleep(0.01)
    acc.stop()

    metrics = acc.compute_metrics()
    # counts
    assert metrics["number_of_images"] == 3
    assert metrics["images_created_count"] == 3

    # initial scores: [1.0, 0.5]
    assert math.isclose(metrics["average_initial_score"], (1.0 + 0.5) / 2)
    assert metrics["min_initial_score"] == 0.5
    assert metrics["max_initial_score"] == 1.0

    # final scores: [2.0, 3.0]
    assert math.isclose(metrics["average_final_score"], (2.0 + 3.0) / 2)
    assert metrics["min_final_score"] == 2.0
    assert metrics["max_final_score"] == 3.0

    # changes: only one change recorded (2.0 - 1.0 = 1.0)
    assert math.isclose(metrics["average_change"], 1.0)
    assert metrics["min_change"] == 1.0
    assert metrics["max_change"] == 1.0

    # durations should be set and avg_image_duration_seconds roughly duration/3
    assert metrics["total_batch_duration_seconds"] is not None
    assert metrics["avg_image_duration_seconds"] is not None
    assert metrics["avg_image_duration_seconds"] <= metrics["total_batch_duration_seconds"]


def test_reset_clears_state():
    acc = BatchImageMetricAccumulator()
    acc.start(step=2)
    acc.add_image(1.2, 1.5)
    acc.stop()
    # metrics non-empty
    metrics = acc.compute_metrics()
    assert metrics["number_of_images"] == 1

    # reset and recompute
    acc.reset()
    metrics2 = acc.compute_metrics()
    assert metrics2["number_of_images"] == 0
    assert metrics2["average_initial_score"] is None
    assert metrics2["average_final_score"] is None
