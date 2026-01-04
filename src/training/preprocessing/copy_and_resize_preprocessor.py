import logging
import os
import random
from pathlib import Path
from typing import Optional

import mlflow
from PIL import Image
from torch.utils.data import DataLoader

from dataset.ImagePathDataset import ImagePathDataset
from dataset.Utils import Utils
from training.hyperparameter_registry import HyperparameterRegistry
from training.mlflow_utils import mlflow_logging
from training.preprocessing.abstract_preprocessor import AbstractPreprocessor
from training.rl_training.training_params import ImagePreprocessingParams

logger = logging.getLogger(__name__)


class CopyAndResizeResult:
    def __init__(self, effective_destination_path: Path, effective_images_path: Path):
        self.effective_destination_path = effective_destination_path
        self.effective_images_path = effective_images_path


class CopyAndResizePreprocessor(AbstractPreprocessor[CopyAndResizeResult]):
    def __init__(self, preprocessing_step: int, dataset_id: Optional[str],
                 source_path: Optional[Path], destination_path: Path):
        super().__init__(preprocessing_step)
        image_preprocessing_params: ImagePreprocessingParams = HyperparameterRegistry.get_store(
            ImagePreprocessingParams).get()
        self.dataset_id = dataset_id
        self.source_path = source_path
        self.image_volume_path = self.get_image_root_path()
        self.destination_path = destination_path
        self.target_max_size = image_preprocessing_params["resize_max_size"]
        self.batch_size = image_preprocessing_params["batch_size"]
        self.max_images = image_preprocessing_params.get("max_images", None)

    @mlflow_logging("perf/copy_and_resize_preprocessor_duration_seconds")
    def _preprocess_impl(self):
        self.effective_destination_path = self.image_volume_path / "copyandresize" / mlflow.active_run().info.experiment_id / mlflow.active_run().info.run_id / self.destination_path
        self.effective_images_path = self.effective_destination_path / "images"
        effective_source_path: Optional[Path] = self.source_path

        if self.dataset_id:
            dataset_config = Utils.get_dataset_config(self.dataset_id)
            effective_source_path = dataset_config.calculate_images_root_path()

        logger.info("Copying and resizing images from %s to %s", effective_source_path, self.effective_images_path)

        dataset: ImagePathDataset = ImagePathDataset(effective_source_path)

        if self.max_images is not None and 0 < self.max_images < len(dataset):
            logger.info(f"Randomly sampling {self.max_images} images from {len(dataset)} available images.")
            indices = random.sample(range(len(dataset)), self.max_images)
            dataset.image_files = [dataset.image_files[i] for i in indices]

        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Utils.collate_keep_size)

        os.makedirs(self.effective_images_path, exist_ok=True)

        for batch in dataloader:
            for image_path, parent_path, filename in batch:
                with Image.open(image_path) as im:
                    im = self._ensure_rgb(im)
                    im.thumbnail((self.target_max_size, self.target_max_size), Image.Resampling.LANCZOS)
                    png_filename = filename.replace(".jpg", ".png")
                    im.save(self.effective_images_path / png_filename)
                    # resized = im.resize((self.target_max_size, self.target_max_size))
                    # resized.save(self.effective_images_path / png_filename)

    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

    def get_preprocessing_result(self) -> CopyAndResizeResult:
        return CopyAndResizeResult(self.effective_destination_path, self.effective_images_path)
