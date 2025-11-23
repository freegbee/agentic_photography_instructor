from typing import Optional

from experiments.shared.BatchImageMetricAccumulator import BatchImageMetricAccumulator


class BatchImageDegradationMetricAccumulator(BatchImageMetricAccumulator):
    """Backward-compatible degradation accumulator that reuses the generic BatchImageMetricAccumulator.

    Keeps the old public API: add_score(score, initial_score) and compute_metrics() returning
    keys like 'average_score', 'min_score', 'max_score' while delegating storage and computation
    to the generic accumulator (which uses final/initial/change terminology).
    """

    def __init__(self):
        super().__init__()

    def add_score(self, score: float, initial_score: Optional[float] = None):
        # Map previous signature (score, initial_score) -> add_image(initial_score, final_score=score)
        return self.add_image(initial_score, score)

    def compute_metrics(self) -> dict:
        # Get generic metrics
        m = super().compute_metrics()
        # Map generic final_score keys to the older names used by ImageDegradationExperiment
        return {
            "step": m.get("step"),
            "number_of_images": m.get("number_of_images"),
            "average_score": m.get("average_final_score"),
            "min_score": m.get("min_final_score"),
            "max_score": m.get("max_final_score"),
            "average_initial_score": m.get("average_initial_score"),
            "min_initial_score": m.get("min_initial_score"),
            "max_initial_score": m.get("max_initial_score"),
            "average_change": m.get("average_change"),
            "min_change": m.get("min_change"),
            "max_change": m.get("max_change"),
            "total_batch_duration_seconds": m.get("total_batch_duration_seconds"),
            "avg_image_duration_seconds": m.get("avg_image_duration_seconds"),
        }
