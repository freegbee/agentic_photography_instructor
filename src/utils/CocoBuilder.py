import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union

from dataset import Utils


class CocoBuilder:
    """
    Minimaler COCO Builder.
    - set_categories: Liste von Kategorienamen setzen (oder beim CSV automatisch erzeugen)
    - save: schreibt JSON
    """

    def __init__(self, dataset_id: Optional[str] = None, source_path: Optional[Path] = None):
        self.info = {
            "description": f"Coco for Dataset {dataset_id}" if dataset_id else f"Coco Dataset for source path {str(source_path)}",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        }
        self.licenses = []
        self.images: List[Dict] = []
        self.annotations: List[Dict] = []
        self.categories: List[Dict] = []
        self._image_id_map: Dict[str, int] = {}
        self._next_image_id = 1
        self._next_ann_id = 1
        self._category_name_to_id: Dict[str, int] = {}
        # sequence counter per image_id for ordered annotations
        self._image_sequence_counters: Dict[int, int] = {}

    def set_description(self, description: str):
        self.info["description"] = description

    def add_image(self, file_name: str, width: int, height: int, score: Optional[float] = None, initial_score: Optional[float] = None) -> int:
        if file_name in self._image_id_map:
            return self._image_id_map[file_name]
        image_id = self._next_image_id
        self._next_image_id += 1
        image_info: Dict[str, Union[int, str, float]] = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        }
        if score is not None:
            image_info["score"] = score
        if initial_score is not None:
            image_info["initial_score"] = initial_score
        self.images.append(image_info)
        self._image_id_map[file_name] = image_id
        return image_id

    def add_image_score_annotation(self, image_id: int, score: float, initial_score: Optional[float] = None) -> int:
        """
        Fügt eine Annotation mit einem Score für das gegebene Bild hinzu.
        - Für image-level scores werden bbox/segmentation leer gelassen und area=0 gesetzt.
        """
        return self._add_image_score_annotation(image_id=image_id, score=score, initial_score=initial_score)

    def add_image_transformation_score_annotation(self, image_id: int, score: float, initial_score: float, transformer_name: Optional[str] = None) -> int:
        """
        Fügt eine Annotation mit einem Score und einem initial_score für das gegebene Bild hinzu.
        """
        return self._add_image_score_annotation(image_id=image_id, score=score, initial_score=initial_score, transformer_name=transformer_name)

    def add_image_transformation_annotation(self, image_id: int, transformer_name: str):
        """
        Fügt eine Annotation für eine Transformation ohne Score hinzu.
        """
        annotation_builder = CocoScoreAnnotationBuilder().with_id(self._next_ann_id).with_image_id(image_id)
        super_category_id = self.ensure_transformer_supercategory()
        if transformer_name not in self._category_name_to_id:
            self.add_category(transformer_name, None, super_category_id)
        cat_id = self._category_name_to_id[transformer_name]
        annotation_builder.with_category_id(cat_id)

        # set sequence for this transformation annotation
        seq = self._get_next_sequence_for_image(image_id)
        annotation_builder.with_sequence(seq)

        ann = annotation_builder.build()
        self.annotations.append(ann)
        self._next_ann_id += 1
        return ann["id"]

    def _add_image_score_annotation(
            self,
            image_id: int,
            score: float,
            initial_score: Optional[float] = None,
            transformer_name: Optional[str] = None
    ) -> int:
        """
        Gemeinsame Implementierung zum Erstellen und Hinzufügen einer Score-Annotation.
        - entweder category_name oder transformer_name optional setzen.
        - initial_score optional für Transformationen.
        """
        if image_id not in {v for v in self._image_id_map.values()}:
            raise ValueError(f"image_id {image_id} not known")

        annotation_builder = (CocoScoreAnnotationBuilder()
                              .with_id(self._next_ann_id)
                              .with_image_id(image_id)
                              .with_score(score)
                              .with_category_id(None))

        if initial_score is not None:
            annotation_builder.with_initial_score(initial_score)

        # If a transformer_name is provided, ensure its category exists and set category_id and transformation label
        if transformer_name is not None:
            super_category_id = self.ensure_transformer_supercategory()
            if transformer_name not in self._category_name_to_id:
                self.add_category(transformer_name, None, super_category_id)
            cat_id = self._category_name_to_id[transformer_name]
            annotation_builder.with_category_id(cat_id)
            annotation_builder.with_transformation(transformer_name)

        # set sequence for this annotation (increment per image)
        seq = self._get_next_sequence_for_image(image_id)
        annotation_builder.with_sequence(seq)

        ann = annotation_builder.build()
        self.annotations.append(ann)
        self._next_ann_id += 1
        return ann["id"]

    def add_category(self, category_name: str, category_id: Optional[int] = None,
                     super_category_id: Optional[int] = None) -> int:
        """
        Fügt eine Kategorie hinzu und gibt deren ID zurück.
        - Wenn category_id None ist, wird eine neue ID automatisch vergeben.
        - Wenn die Kategorie bereits existiert, wird die vorhandene ID zurückgegeben.
        - super_category_id kann gesetzt werden, um eine Superkategorie zuzuweisen.
        """
        if category_name in self._category_name_to_id:
            return self._category_name_to_id[category_name]

        existing_ids = {c["id"] for c in self.categories}
        if category_id is None:
            new_id = (max(existing_ids) + 1) if existing_ids else 1
        else:
            if category_id in existing_ids:
                raise ValueError(f"category_id {category_id} already used")
            new_id = category_id

        cat = {"id": new_id, "name": category_name, "supercategory": super_category_id}
        self.categories.append(cat)
        self._category_name_to_id[category_name] = new_id
        return new_id

    def ensure_transformer_supercategory(self) -> int:
        """
        Fügt die Superkategorie 'transformer' hinzu und gibt deren ID zurück.

        Convenience-Methode für Transformer-Kategorie um naming sicherzustellen
        """
        category_name = Utils.TRANSFORMER_CATEGORY_NAME
        return self.add_category(category_name)

    def save(self, target_filename: str):
        coco = {
            "info": self.info,
            "licenses": self.licenses,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }

        dir_name = os.path.dirname(target_filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(target_filename, "w", encoding="utf-8") as fh:
            json.dump(coco, fh, ensure_ascii=False, indent=2)

    def to_json_string(self) -> str:
        coco = {
            "info": self.info,
            "licenses": self.licenses,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        return json.dumps(coco, ensure_ascii=False, indent=2)

    def _get_next_sequence_for_image(self, image_id: int) -> int:
        """Return the next sequence number for `image_id` and increment the internal counter."""
        if image_id not in self._image_sequence_counters:
            self._image_sequence_counters[image_id] = 1
            return 1
        else:
            self._image_sequence_counters[image_id] += 1
            return self._image_sequence_counters[image_id]

    def add_image_final_score_annotation(self, image_id: int, score: float, initial_score: Optional[float] = None) -> int:
        """Add an image-level score annotation WITHOUT sequence and with category_id set to 0.

        This is intended to log the overall initial and final score for the image after
        all transformations. It deliberately does NOT set a sequence value.
        """
        if image_id not in {v for v in self._image_id_map.values()}:
            raise ValueError(f"image_id {image_id} not known")

        annotation_builder = (CocoScoreAnnotationBuilder()
                              .with_id(self._next_ann_id)
                              .with_image_id(image_id)
                              .with_score(score)
                              .with_category_id(None))

        if initial_score is not None:
            annotation_builder.with_initial_score(initial_score)

        ann = annotation_builder.build()
        self.annotations.append(ann)
        self._next_ann_id += 1
        return ann["id"]


class CocoScoreAnnotationBuilder:
    """
    Hilfsklasse zum Erstellen von COCO Score Annotations.
    """

    def __init__(self):
        self.id = None
        self.image_id: Optional[int] = None
        self.score: Optional[float] = None
        self.initial_score: Optional[float] = None
        self.category_id: int = 0
        self.sequence: Optional[int] = None

    def with_id(self, ann_id: int) -> 'CocoScoreAnnotationBuilder':
        self.id = ann_id
        return self

    def with_image_id(self, image_id: int) -> 'CocoScoreAnnotationBuilder':
        self.image_id = image_id
        return self

    def with_score(self, score: float) -> 'CocoScoreAnnotationBuilder':
        self.score = score
        return self

    def with_initial_score(self, initial_score: float) -> 'CocoScoreAnnotationBuilder':
        self.initial_score = initial_score
        return self

    def with_category_id(self, category_id: Union[int, None]) -> 'CocoScoreAnnotationBuilder':
        self.category_id = 0 if category_id is None else category_id
        return self

    def with_sequence(self, sequence: int) -> 'CocoScoreAnnotationBuilder':
        self.sequence = sequence
        return self

    def with_transformation(self, transformation: str) -> 'CocoScoreAnnotationBuilder':
        self.transformation = transformation
        return self

    def build(self) -> Dict:
        if self.id is None:
            raise ValueError("Annotation ID must be set")

        if self.image_id is None:
            raise ValueError("Image ID must be set")

        ann: Dict = {
            "id": self.id,  # ID wird vom CocoBuilder gesetzt
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": [],  # image-level: keine bbox
            "area": 0,
            "iscrowd": 0,
            "segmentation": [],
        }
        if self.sequence is not None:
            ann["sequence"] = self.sequence
        if self.score is not None:
            ann["score"] = self.score
        if self.initial_score is not None:
            ann["initial_score"] = self.initial_score
        if getattr(self, 'transformation', None) is not None:
            ann["transformation"] = self.transformation
        return ann
