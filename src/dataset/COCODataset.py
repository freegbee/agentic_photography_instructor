import logging
from pathlib import Path
from typing import List, TypedDict

from numpy import ndarray
from torch.utils.data import Dataset

from data_types.AgenticImage import ImageData
from dataset.agentic_coco_image import CocoImageData
from dataset.enhanced_coco import EnhancedCOCO

logger = logging.getLogger(__name__)


class TransformationAnnotationTypeDict(TypedDict):
    sequence: int
    category_id: int
    transformation_label: str


class COCODataset(Dataset[CocoImageData]):
    """
    Custom Dataset for COCO formatted data.
    Implementierung gemäss https://www.codegenes.net/blog/load-coco-data-pytorch/

    Args:
        images_root_path (Path): Verzeichnis mit den Bildern.
        annotation_file (Path): Pfad zur COCO Annotationsdatei.json.
    """

    def __init__(self, images_root_path: Path, annotation_file: Path, score_category_id: int = 0,
                 transformation_supercategory_id: int = 2):
        self.images_root_path: Path = images_root_path
        self.coco: EnhancedCOCO = EnhancedCOCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.score_category_id = score_category_id
        self.transformation_supercategory_id = self.coco.getCatIds(catNms=["transformer"])[0]
        # Ja, es ist ein Bug: supNums ist eine Liste (oder eine einzelne) ID einer Super-Categorie
        self.transformation_cats = self.coco.getCatIds(supNms=self.transformation_supercategory_id)
        self.scores = []

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx) -> ImageData:
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = self.images_root_path / img_info['file_name']
        image_data, image_color_order = self._load_image(image_path)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        score = 0.0
        initial_score = 0.0

        transformations: List[TransformationAnnotationTypeDict] = []

        for ann in anns:
            if ann['category_id'] == self.score_category_id:
                score = ann.get("score", 0.0)
                initial_score = ann.get("initial_score", 0.0)
            elif ann['category_id'] in self.transformation_cats:
                transformations.append(TransformationAnnotationTypeDict({
                    'sequence': ann['sequence'],
                    'category_id': ann['category_id'],
                    'transformation_label': self.coco.cats[ann['category_id']]['name']
                }))

        transformations = sorted(transformations, key=lambda t: t.get("sequence"))

        result = CocoImageData(id=image_id,
                               image_path=image_path,
                               image_relative_path=Path(img_info['file_name']),
                               width=img_info['width'],
                               height=img_info['height'],
                               image_data=image_data,
                               image_color_order=image_color_order,
                               score=score,
                               annotations=self.coco.imgToAnns[image_id],
                               initial_score=initial_score,
                               applied_transformations=[t.get("transformation_label") for t in transformations])

        return result

    @staticmethod
    def _load_image(image_path: Path):
        import cv2
        image: ndarray = cv2.imread(str(image_path))
        return image, "BGR"

    def set_scores(self, scores):
        self.scores = scores
