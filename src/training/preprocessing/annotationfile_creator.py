import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dataset.Utils import Utils
from image_acquisition.AbsolutePathImageDataset import AbsolutePathImageDataset
from training import mlflow_helper
from training.hyperparameter_registry import HyperparameterRegistry
from training.preprocessing.abstract_preprocessor import AbstractPreprocessor
from training.rl_training.training_params import GeneralPreprocessingParams
from utils.CocoBuilder import CocoBuilder

logger = logging.getLogger(__name__)


class AnnotationFileCreatorResponse:
    def __init__(self, annotations_file_path: Path):
        self.annotations_file_path = annotations_file_path


class AnnotationFileCreator(AbstractPreprocessor[AnnotationFileCreatorResponse]):
    """Preprocessor for creating annotation files for datasets without existing annotations."""

    def __init__(self, preprocessing_step: int):
        super().__init__(preprocessing_step)
        general_params = HyperparameterRegistry.get_store(GeneralPreprocessingParams).get()
        self.batch_size = general_params["batch_size"]
        self.source_path: Optional[Path] = None
        self.images_path: Optional[Path] = None
        self.annotations_file_path: Optional[Path] = None

    def with_source_path(self, source_path: Path) -> 'AnnotationFileCreator':
        self.source_path = source_path
        self.annotations_file_path = self.source_path / 'annotations.json'
        return self

    def with_image_path(self, images_path: Path) -> 'AnnotationFileCreator':
        self.images_path = images_path
        return self

    def _preprocess_impl(self):
        self.coco_builder = CocoBuilder(source_path=self.source_path)
        self._ingest_files()

        # Nachverarbeitung
        self.coco_builder.save(str(self.annotations_file_path))

        mlflow_helper.log_artifact(local_path=str(self.annotations_file_path), step=self.step_arg, artifact_path="initial_annotation")
        mlflow_helper.log_metric("total_number_of_images", len(self.coco_builder.images), self.step_arg)

    def _ingest_files(self):
        logger.info("Start building annotation file for images in path %s", self.images_path)
        dataset: Dataset = AbsolutePathImageDataset(self.images_path)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Utils.collate_keep_size)
        for batch_index, batch in enumerate(dataloader):
            self._process_batch(batch)

    def _process_batch(self, batch):
        for image_full_path, path, file_name in batch:
            logger.debug("Process single file path %s", image_full_path)
            with Image.open(image_full_path) as image:
                width, height = image.size
                self.coco_builder.add_image(file_name, width, height)

    def get_preprocessing_result(self) -> AnnotationFileCreatorResponse:
        return AnnotationFileCreatorResponse(self.annotations_file_path)
