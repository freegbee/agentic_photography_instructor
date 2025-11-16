from pathlib import Path

from numpy import ndarray
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from data_types.AgenticImage import ImageData


class COCODataset(Dataset[ImageData]):
    """
    Custom Dataset for COCO formatted data.
    Implementierung gemäss https://www.codegenes.net/blog/load-coco-data-pytorch/

    Args:
        root_dir (Path): Verzeichnis mit den Bildern.
        annotation_file (Path): Pfad zur COCO Annotationsdatei.json.
    """

    def __init__(self, root_dir: Path, annotation_file: Path):
        self.root_dir: Path = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx) -> ImageData:
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = self.root_dir / img_info['file_name']
        image_data, image_color_order = self._load_image(image_path)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        score = 0.0
        for ann in anns:
            if ann['category_id'] == 0:
                score = ann.get("score", 0.0)
                break

        result = ImageData(image_path=image_path,
                           image_relative_path=Path(img_info['file_name']),
                           image_data=image_data,
                           image_color_order=image_color_order,
                           score=score)

        return result


    @staticmethod
    def _load_image(image_path: Path):
        import cv2
        image: ndarray = cv2.imread(str(image_path))
        return image, "BGR"
