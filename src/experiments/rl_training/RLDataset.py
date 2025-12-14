import logging
from pathlib import Path
from typing import Tuple

from numpy import ndarray
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from data_types.AgenticImage import ImageData

logger = logging.getLogger(__name__)


class RLDataset(Dataset):
    """
    Dataset for RL training that loads degraded images along with their original counterparts.

    Each sample contains:
    - degraded_image: The degraded (starting) image
    - degraded_score: Aesthetic score of the degraded image
    - original_image: The original (target) image
    - original_score: Aesthetic score of the original image (target)
    - transformation: The transformation that was applied to create the degraded version
    - image_id: Unique identifier
    - image_path: Relative path to the degraded image

    The COCO annotations must contain:
    - 'score' annotation for the degraded image score
    - 'original_score' annotation for the target score
    - 'transformation' annotation indicating which transformer was applied
    """

    def __init__(self, degraded_images_root: Path, annotation_file: Path, score_category_id: int = 0):
        """
        Args:
            degraded_images_root: Directory containing the degraded images
            annotation_file: Path to COCO format annotation file
            score_category_id: Category ID for score annotations (default 0)
        """
        self.degraded_images_root = degraded_images_root
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.score_category_id = score_category_id

        logger.info("Initialized RLDataset with %d images from %s", len(self.image_ids), degraded_images_root)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx) -> Tuple[ImageData, float, str]:
        """
        Returns:
            Tuple of (degraded_image_data, original_score, transformation_name)
        """
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = self.degraded_images_root / img_info['file_name']

        # Load degraded image
        degraded_image, color_order = self._load_image(image_path)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        degraded_score = 0.0
        initial_score = 0.0
        transformation = "unknown"

        for ann in anns:
            if ann['category_id'] == self.score_category_id:
                degraded_score = ann.get("score", 0.0)
                initial_score = ann.get("initial_score", 0.0)
            # Look for transformation annotation
            if 'transformation' in ann:
                transformation = ann['transformation']

        # Create ImageData for the degraded image
        degraded_image_data = ImageData(
            id=image_id,
            image_path=image_path,
            image_relative_path=Path(img_info['file_name']),
            image_data=degraded_image,
            image_color_order=color_order,
            score=degraded_score
        )

        return degraded_image_data, initial_score, transformation

    @staticmethod
    def _load_image(image_path: Path) -> Tuple[ndarray, str]:
        """Load image using OpenCV in BGR format."""
        import cv2
        image: ndarray = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        return image, "BGR"
