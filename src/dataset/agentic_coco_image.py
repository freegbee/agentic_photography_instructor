from typing import List, Optional

from data_types.AgenticImage import ImageData
from dataset.enhanced_coco import _EnhancedAnnotation


class CocoImageData(ImageData):
    initial_score: float
    width: int
    height: int
    annotations: Optional[List[_EnhancedAnnotation]]

    def clone(self):
        cloned = super().clone()
        cloned.width = self.width
        cloned.height = self.height
        cloned.initial_score = self.initial_score
        cloned.annotations = self.annotations.copy() if self.annotations else None
        return cloned

    def with_initial_score(self, initial_score: float) -> 'CocoImageData':
        self.initial_score = initial_score
        return self

    def with_annotations(self, annotations: List[_EnhancedAnnotation]) -> 'CocoImageData':
        self.annotations = annotations
        return self
