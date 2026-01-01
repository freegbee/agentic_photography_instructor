import logging
import random
from pathlib import Path
from typing import Optional, Dict, List

import mlflow
from torch.utils.data import Sampler, DataLoader

from dataset.COCODataset import COCODataset
from dataset.Utils import Utils
from dataset.enhanced_coco import AnnotationFileAndImagePath
from dataset.samplers import IndexSampler
from training import mlflow_helper
from training.mlflow_utils import mlflow_logging
from training.preprocessing.abstract_preprocessor import AbstractPreprocessor
from training.split_ratios import SplitRatios
from utils.CocoBuilder import CocoBuilder
from utils.ImageUtils import ImageUtils

logger = logging.getLogger(__name__)

class SplitPreprocessorResult:
    def __init__(self, effective_destination_root_path: Path, processed_splits: Dict[str, AnnotationFileAndImagePath]):
        self.effective_destination_root_path = effective_destination_root_path
        self.processed_splits: Dict[str, AnnotationFileAndImagePath] = processed_splits


class SplitPreprocessor(AbstractPreprocessor[SplitPreprocessorResult]):

    def __init__(self, preprocessing_step: Optional[int]):
        super().__init__(preprocessing_step)
        self.image_volume_path = self.get_image_root_path()
        self.source_path: Optional[Path] = None
        self.source_annotation_file: Optional[Path] = None
        self.destination_root_path: Optional[Path] = None
        self.random_generator: Optional[random.Random] = None
        self.split_ratios: Optional[SplitRatios] = None
        self.batch_size: int = 1
        self.processed_splits: Dict[str, AnnotationFileAndImagePath] = {}


    def with_source_path(self, source_path: Path) -> 'SplitPreprocessor':
        self.source_path = source_path
        return self

    def with_destination_root_path(self, destination_root_path: Path) -> 'SplitPreprocessor':
        self.destination_root_path = destination_root_path
        return self

    def with_source_annotation_file(self, source_annotation_file: Path) -> 'SplitPreprocessor':
        self.source_annotation_file = source_annotation_file
        return self

    def with_split_ratios(self, split_ratios: SplitRatios):
        self.split_ratios = split_ratios
        return self

    def with_batch_size(self, batch_size: int) -> 'SplitPreprocessor':
        self.batch_size = batch_size
        return self

    def with_random_seed(self, seed: int) -> 'SplitPreprocessor':
        self.random_generator = random.Random(seed)
        return self

    @mlflow_logging("image_split_duration_seconds")
    def _preprocess_impl(self):
        self.effective_destination_root_path = self.image_volume_path / "split" / mlflow.active_run().info.experiment_id / mlflow.active_run().info.run_id / self.destination_root_path
        source_dataset: COCODataset = COCODataset(self.source_path, self.source_annotation_file, include_transformations=False)
        splits = self._create_splits(source_dataset)
        for split_name, split_indices in splits.items():
            logger.info(f"Generating {split_name} split with {len(split_indices)} images")

            split_dir = self.effective_destination_root_path / split_name
            split_images_dir = split_dir / "images"
            split_coco_builder = CocoBuilder(source_path=self.effective_destination_root_path)
            split_coco_builder.set_description(
                f"Coco file for {split_name} split from source path {self.effective_destination_root_path}"
            )
            index_sampler = IndexSampler(split_indices)
            self._split_by_index(split_coco_builder, source_dataset, index_sampler, split_images_dir)

            split_coco_file = split_dir / "annotations.json"
            split_coco_builder.save(str(split_coco_file))
            mlflow_helper.log_artifact(local_path=str(split_coco_file), step=self.step_arg,
                                       artifact_path=str(split_name))

            mlflow_helper.log_metric(f"{split_name}_num_images", len(split_coco_builder.images))
            self.processed_splits[split_name] = AnnotationFileAndImagePath(split_coco_file, split_images_dir)
            logger.info("Finished split %s with %d images", split_name, len(split_indices))

        logger.info("Finished generating all splits")


    def _split_by_index(self, coco_builder: CocoBuilder, source_dataset: COCODataset, sampler: Sampler,
                          target_directory_root: Path):
        dataloader = DataLoader(source_dataset, batch_size=self.batch_size, sampler=sampler,
                                collate_fn=Utils.collate_keep_size)

        for batch_index, batch in enumerate(dataloader):
            for image_data in batch:
                split_image = image_data.get_image_data("BGR")
                copied_image_path = str(target_directory_root / image_data.image_relative_path)
                ImageUtils.save_image(split_image, copied_image_path)
                image_id = coco_builder.add_image(file_name=str(image_data.image_relative_path),
                                                  width=image_data.width,
                                                  height=image_data.height,
                                                  initial_score=image_data.initial_score)
                score = image_data.score if image_data.score else image_data.initial_score
                coco_builder.add_image_score_annotation(image_id=image_id, score=score, initial_score=image_data.initial_score)
            logger.info(f"Finished processing batch {batch_index} with {len(batch)} images")

    def _create_splits(self, source_dataset: COCODataset) -> Dict[str, List[int]]:
        total_images = len(source_dataset)
        indices = list(range(total_images))
        self.random_generator.shuffle(indices)
        return self.split_ratios.get_split_indices(indices)

    def get_preprocessing_result(self) -> SplitPreprocessorResult:
        return SplitPreprocessorResult(self.effective_destination_root_path, self.processed_splits)