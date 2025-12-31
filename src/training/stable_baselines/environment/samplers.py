import logging
from abc import ABC
from typing import Tuple

import numpy as np

from data_types.AgenticImage import ImageData
from dataset.COCODataset import COCODataset

logger = logging.getLogger(__name__)


class CocoDatasetSampler(ABC):
    def __init__(self, dataset: COCODataset):
        self.dataset = dataset
        self.dataset_size = len(self.dataset)

    def __call__(self) -> Tuple[int, ImageData, bool]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reset(self):
        raise NotImplementedError("This method should be implemented by subclasses if supporting reset.")


class RandomCocoDatasetSampler(CocoDatasetSampler):
    def __init__(self, dataset: COCODataset, seed: int):
        super().__init__(dataset)
        self.rng = np.random.RandomState(seed)

    def __call__(self) -> Tuple[int, ImageData, bool]:
        idx = int(self.rng.randint(0, self.dataset_size))
        return idx, self.dataset[idx], False


class SequentialCocoDatasetSampler(CocoDatasetSampler):
    def __init__(self, dataset: COCODataset):
        super().__init__(dataset)
        self.current_index = 0

    def __call__(self) -> Tuple[int, ImageData, bool]:
        logger.debug(f"Sequential sampler at index {self.current_index}/{self.dataset_size}")
        idx = self.current_index
        image_data = self.dataset[idx]
        self.current_index += 1
        return idx, image_data, self.current_index >= self.dataset_size

    def reset(self):
        self.current_index = 0
