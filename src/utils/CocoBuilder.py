import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class CocoBuilder:
    """
    Minimaler COCO Builder.
    - set_categories: Liste von Kategorienamen setzen (oder beim CSV automatisch erzeugen)
    - save: schreibt JSON
    """
    def __init__(self, dataset_id):
        self.info = {
            "description": f"Coco for Dataset {dataset_id}",
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

    def set_description(self, description: str):
        self.info["description"] = description

    def add_image(self, file_name: str, width: int, height: int) -> int:
        if file_name in self._image_id_map:
            return self._image_id_map[file_name]
        image_id = self._next_image_id
        self._next_image_id += 1
        image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        }
        self.images.append(image_info)
        self._image_id_map[file_name] = image_id
        return image_id

    def add_image_score_annotation(self, image_id: int, score: float, category_name: Optional[str] = None) -> int:
        """
        Fügt eine Annotation mit einem Score für das gegebene Bild hinzu.
        - Wenn category_name gesetzt ist, wird die Kategorie sichergestellt/angelegt.
        - Für image-level scores werden bbox/segmentation leer gelassen und area=0 gesetzt.
        """
        # optional: sicherstellen, dass image_id existiert
        if image_id not in {v for v in self._image_id_map.values()}:
            raise ValueError(f"image_id {image_id} not known")

        # Kategorie anlegen/finden falls angegeben
        cat_id = None
        if category_name:
            if category_name not in self._category_name_to_id:
                new_id = len(self.categories) + 1
                self.add_category(category_name, new_id)
            cat_id = self._category_name_to_id[category_name]

        ann = {
            "id": self._next_ann_id,
            "image_id": image_id,
            "category_id": cat_id if cat_id is not None else 0,
            "bbox": [],  # image-level: keine bbox
            "area": 0,
            "iscrowd": 0,
            "segmentation": [],
            "score": float(score)  # hier wird der Score gespeichert
        }

        self.annotations.append(ann)
        self._next_ann_id += 1
        return ann["id"]

    def add_image_transformation_score_annotation(self, image_id: int, score: float, initial_score: float, transformer_name: str, super_category_id: int) -> int:
        """
        Fügt eine Annotation mit einem Score für das gegebene Bild hinzu.
        - Wenn category_name gesetzt ist, wird die Kategorie sichergestellt/angelegt.
        - Für image-level scores werden bbox/segmentation leer gelassen und area=0 gesetzt.
        """
        # optional: sicherstellen, dass image_id existiert
        if image_id not in {v for v in self._image_id_map.values()}:
            raise ValueError(f"image_id {image_id} not known")

        # transformer_name als Kategorie anlegen/finden falls angegeben
        cat_id = None
        if transformer_name:
            if transformer_name not in self._category_name_to_id:
                new_id = len(self.categories) + 1
                self.add_category(transformer_name, new_id, super_category_id)
            cat_id = self._category_name_to_id[transformer_name]

        ann = {
            "id": self._next_ann_id,
            "image_id": image_id,
            "category_id": cat_id if cat_id is not None else 0,
            "bbox": [],  # image-level: keine bbox
            "area": 0,
            "iscrowd": 0,
            "segmentation": [],
            "score": float(score),  # hier wird der Score gespeichert
            "initial_score": float(initial_score),  # hier wird der Score gespeichert
        }

        self.annotations.append(ann)
        self._next_ann_id += 1
        return ann["id"]

    def add_category(self, category_name: str, category_id: Optional[int] = None, super_category_id: Optional[int] = None) -> int:
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