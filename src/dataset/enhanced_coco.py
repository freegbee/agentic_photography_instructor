from pathlib import Path
from typing import TypedDict, Optional

from pycocotools.coco import COCO


class _EnhancedAnnotation(TypedDict):
    """Enhanced Annotation class with additional fields."""
    id: int
    image_id: int
    category_id: int
    segmentation: list
    area: float
    bbox: list[float]
    iscrowd: int
    is_scoring_annotation: bool
    score: float
    initial_score: float


class EnhancedCOCO(COCO):
    """Enhanced COCO class with additional functionality."""

    def __init__(self, annotation_file=None):
        super(EnhancedCOCO, self).__init__(annotation_file)
        self.anns: dict[int, _EnhancedAnnotation] = self.anns  # type: ignore
        self.imgToAnns: dict[int, list[_EnhancedAnnotation]] = self.imgToAnns # type: ignore


    def update_annotation_score(self, annotation_id: int, score: float):
        """Update a specific key-value pair in an annotation."""
        self.anns[annotation_id]["score"] = score

    def update_annotation_initial_score(self, annotation_id: int, initial_score: float):
        """Update a specific key-value pair in an annotation."""
        self.anns[annotation_id]["initial_score"] = initial_score

    def get_annotations_by_image_id(self, image_id: int) -> list[_EnhancedAnnotation]:
        """Get annotations for a specific image ID."""
        return self.imgToAnns.get(image_id, [])


    @staticmethod
    def get_root_annotation(annotations: list[_EnhancedAnnotation]) -> _EnhancedAnnotation:
        """Get the root annotation from a list of annotations."""
        for ann in annotations:
            if ann.get("is_scoring_annotation", False):
                return ann
        raise ValueError("No root annotation found.")

    def add_scoring_annotation(self, image_id: int, score: float, initial_score: float,  category_id: Optional[int] = 0) -> int:
        """Add a scoring annotation to the COCO dataset."""
        annotation_id = max(self.anns.keys(), default=0) + 1
        annotation: _EnhancedAnnotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id if category_id is not None else 0,
            "segmentation": [],
            "area": 0.0,
            "bbox": [],
            "iscrowd": 0,
            "is_scoring_annotation": True,
            "score": score,
            "initial_score": initial_score
        }
        self.anns[annotation_id] = annotation
        if image_id not in self.imgToAnns:
            self.imgToAnns[image_id] = []
        self.imgToAnns[image_id].append(annotation)
        return annotation_id


class AnnotationFileAndImagePath:
    def __init__(self, annotation_file: Path, images_path: Path):
        self.annotation_file = annotation_file
        self.images_path = images_path
