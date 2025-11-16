import time
from typing import List


class BatchImageDegradationMetricAccumulator:
    def __init__(self):
        self.step = None
        self.start_time = None
        self.end_time = None
        self.scores: List[float] = []

    def start(self, step: int = 0):
        self.step = step
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def add_score(self, score: float):
        self.scores.append(score)

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.scores = []

    def compute_metrics(self) -> dict:
        duration = (self.end_time - self.start_time) if self.start_time and self.end_time else None
        return {
            "step": self.step,
            "number_of_images": len(self.scores),
            "total_batch_duration_seconds": duration if duration  else None,
            "avg_image_duration_seconds": duration/len(self.scores) if duration else None
        }