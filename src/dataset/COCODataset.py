from pathlib import Path
from typing import Optional, List

from numpy import ndarray
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from data_types.AgenticImage import ImageData


class COCODataset(Dataset[ImageData]):
    """
    Custom Dataset for COCO formatted data.
    Implementierung gemäss https://www.codegenes.net/blog/load-coco-data-pytorch/

    Args:
        images_root_path (Path): Verzeichnis mit den Bildern.
        annotation_file (Path): Pfad zur COCO Annotationsdatei.json.
    """

    def __init__(self, images_root_path: Path, annotation_file: Path, score_category_id: int = 0):
        self.images_root_path: Path = images_root_path
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.score_category_id = score_category_id
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
        for ann in anns:
            if ann['category_id'] == self.score_category_id:
                score = ann.get("score", 0.0)
                break

        result = ImageData(id=image_id,
                           image_path=image_path,
                           image_relative_path=Path(img_info['file_name']),
                           image_data=image_data,
                           image_color_order=image_color_order,
                           score=score)

        return result

    def get_transformer_descendant_ids(self) -> List[int]:
        cats_iter = self.coco.loadCats()
        transformer_ids = self.coco.getCatIds(catNms=["transformer"])

        def chain_contains_transformer(cat_id: Optional[int]) -> bool:
            visited = set()
            while cat_id is not None and cat_id not in visited:
                visited.add(cat_id)
                if cat_id in transformer_ids:
                    return True
                cat = self.coco.loadCats(cat_id)[0]
                if not cat:
                    break
                cat_id = cat.get("supercategory")
            return False

        self.utils_transformer_descendant_ids = [c["id"] for c in cats_iter if chain_contains_transformer(c.get("id"))]

        return self.utils_transformer_descendant_ids

    @staticmethod
    def _load_image(image_path: Path):
        import cv2
        image: ndarray = cv2.imread(str(image_path))
        return image, "BGR"

    def set_scores(self, scores):
        self.scores = scores
