import time
from typing import List, Optional


class BatchImageMetricAccumulator:
    """Generische Batch-Metrik-Accumulator-Klasse für Bilder.

    Speichert initiale Scores, finale Scores (nach Transformation) und berechnet Änderungen.
    Eignet sich für Logging pro Batch und Aggregation über einen Lauf.
    """

    def __init__(self):
        self.step: Optional[int] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.initial_scores: List[float] = []
        self.final_scores: List[float] = []
        self.changes: List[float] = []
        self.number_of_images: int = 0

    def start(self, step: int = 0):
        self.step = step
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def reset(self):
        self.step = None
        self.start_time = None
        self.end_time = None
        self.initial_scores = []
        self.final_scores = []
        self.changes = []
        self.number_of_images = 0

    def add_image(self, initial_score: Optional[float], final_score: Optional[float]):
        """Fügt ein Bild mit optionalen initialen und finalen Scores hinzu.

        Falls ein Score None ist, wird er nicht in die Liste aufgenommen, aber die Anzahl
        der Bilder wird dennoch hochgezählt (das ermöglicht accurate "images processed" Zählung).
        """
        self.number_of_images += 1
        if initial_score is not None:
            try:
                self.initial_scores.append(float(initial_score))
            except Exception:
                pass
        if final_score is not None:
            try:
                self.final_scores.append(float(final_score))
            except Exception:
                pass
        # compute change only if both present
        if initial_score is not None and final_score is not None:
            try:
                self.changes.append(float(final_score) - float(initial_score))
            except Exception:
                pass

    @staticmethod
    def _safe_stats(values: List[float]):
        if not values:
            return None, None, None
        avg = sum(values) / len(values)
        return avg, min(values), max(values)

    def compute_metrics(self) -> dict:
        duration = (self.end_time - self.start_time) if (self.start_time and self.end_time) else None

        avg_final, min_final, max_final = self._safe_stats(self.final_scores)
        avg_init, min_init, max_init = self._safe_stats(self.initial_scores)
        avg_change, min_change, max_change = self._safe_stats(self.changes)

        return {
            "step": self.step,
            "number_of_images": self.number_of_images,
            "images_created_count": self.number_of_images,
            "average_final_score": avg_final,
            "min_final_score": min_final,
            "max_final_score": max_final,
            "average_initial_score": avg_init,
            "min_initial_score": min_init,
            "max_initial_score": max_init,
            "average_change": avg_change,
            "min_change": min_change,
            "max_change": max_change,
            "total_batch_duration_seconds": duration if duration is not None else None,
            "avg_image_duration_seconds": (duration / self.number_of_images) if (duration is not None and self.number_of_images > 0) else None
        }
