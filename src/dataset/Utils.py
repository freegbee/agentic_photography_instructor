import logging
from pathlib import Path
from typing import List, cast, Optional

from torch.utils.data import DataLoader, Dataset

from data_types.AgenticImage import ImageData
from dataset.COCODataset import COCODataset
from dataset.TopKSampler import TopKSampler

logger = logging.getLogger(__name__)

TRANSFORMER_CATEGORY_NAME = "transformer"
ROOT_CATEGORY_ID = 1


class Utils:
    @staticmethod
    def load_coco_dataset(dataset_root_path: Path) -> COCODataset:
        """
        Lädt ein COCODataset aus dem angegebenen Verzeichnis und der Annotationsdatei.
        """
        return COCODataset(Utils.calculate_images_root_path(dataset_root_path), Utils.calculate_annotations_file_path(dataset_root_path))

    @staticmethod
    def create_topk_coco_dataloader(root_path: Path, batch_size: int, k: int) -> 'DataLoader':
        source_dataset = Utils.load_coco_dataset(root_path)
        Utils.calculate_dataset_scores(source_dataset)
        logger.info("Calculated %d scores for dataset", len(source_dataset.scores))
        sampler = TopKSampler(source_dataset, k=k)
        dataloader = DataLoader(cast(Dataset[ImageData], source_dataset), batch_size=batch_size, sampler=sampler,
                                collate_fn=Utils.collate_keep_size)
        return dataloader

    @staticmethod
    def calculate_annotations_file_path(dataset_root_path: Path) -> Path:
        return dataset_root_path / "annotations.json"

    @staticmethod
    def calculate_images_root_path(dataset_root_path: Path) -> Path:
        return dataset_root_path / "images"

    @staticmethod
    def calculate_dataset_scores(dataset: COCODataset) -> List[float]:
        """
        Berechnet dataset.scores für ein COCODataset ohne __getitem__-Sideeffects.
        Nutzt dataset.coco und dataset.image_ids.
        """
        scores = []
        for image_id in dataset.image_ids:
            ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
            anns = dataset.coco.loadAnns(ann_ids)
            score = 0.0
            for ann in anns:
                if ann.get('category_id') == 0:
                    score = ann.get("score", 0.0)
                    break
            scores.append(score)
        dataset.set_scores(scores)
        return scores

    @staticmethod
    def collate_keep_size(batch):
        # Returns a batch without stacking the images, so that they can keep their original size
        return batch
