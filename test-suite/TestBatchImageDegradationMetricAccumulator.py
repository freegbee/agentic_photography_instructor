# ensure project src is on sys.path for imports when running tests from repo root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import time
import math

from experiments.image_degrade.BatchImageDegradationMetricAccumulator import BatchImageDegradationMetricAccumulator


def test_empty_degradation_accumulator():
    acc = BatchImageDegradationMetricAccumulator()
    acc.reset()
    metrics = acc.compute_metrics()
    assert metrics["number_of_images"] == 0
    assert metrics["average_score"] is None
    assert metrics["average_initial_score"] is None


def test_add_scores_and_compute():
    acc = BatchImageDegradationMetricAccumulator()
    acc.reset()
    acc.start(step=0)
    acc.add_score(2.0, 1.0)
    acc.add_score(3.0, None)
    acc.add_score(None, 0.5)  # allow None final? original API expected float, but be defensive
    time.sleep(0.005)
    acc.stop()

    metrics = acc.compute_metrics()
    assert metrics["number_of_images"] == 3
    # average_score should consider final scores where present -> [2.0,3.0]
    assert math.isclose(metrics["average_score"], (2.0 + 3.0) / 2)
    assert metrics["min_score"] == 2.0
    assert metrics["max_score"] == 3.0

    # initial average: [1.0, None, 0.5] -> only [1.0,0.5]
    assert math.isclose(metrics["average_initial_score"], (1.0 + 0.5) / 2)


def test_reset_behavior():
    acc = BatchImageDegradationMetricAccumulator()
    acc.start(step=5)
    acc.add_score(1.5, 1.0)
    acc.stop()
    assert acc.compute_metrics()["number_of_images"] == 1
    acc.reset()
    assert acc.compute_metrics()["number_of_images"] == 0

