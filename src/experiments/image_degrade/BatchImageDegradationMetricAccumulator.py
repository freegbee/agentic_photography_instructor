import time
from typing import List


class BatchImageDegradationMetricAccumulator:
    def __init__(self):
        self.step = None
        self.start_time = None
        self.end_time = None
        self.scores: List[float] = []
        self.initial_scores: List[float] = []
        self.changes: List[float] = []

    def start(self, step: int = 0):
        self.step = step
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def add_score(self, score: float, initial_score: float = None):
        self.scores.append(score)
        self.initial_scores.append(score)
        self.changes.append(score - initial_score)

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.scores = []
        self.initial_scores = []
        self.changes = []

    @staticmethod
    def _safe_stats(values: List[float]):
        if not values:
            return None, None, None
        avg = sum(values) / len(values)
        return avg, min(values), max(values)

    def compute_metrics(self) -> dict:
        num = len(self.scores)
        duration = (self.end_time - self.start_time) if self.start_time and self.end_time else None

        avg_score, min_score, max_score = self._safe_stats(self.scores)
        avg_init, min_init, max_init = self._safe_stats(self.initial_scores)
        avg_change, min_change, max_change = self._safe_stats(self.changes)

        return {
            "step": self.step,
            "number_of_images": num,
            "average_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "average_initial_score": avg_init,
            "min_initial_score": min_init,
            "max_initial_score": max_init,
            "average_change": avg_change,
            "min_change": min_change,
            "max_change": max_change,
            "total_batch_duration_seconds": duration if duration is not None else None,
            "avg_image_duration_seconds": (duration / num) if (duration is not None and num > 0) else None
        }
