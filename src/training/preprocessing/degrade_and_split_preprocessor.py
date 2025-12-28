import logging
import random
from pathlib import Path
from typing import Optional, List, Dict, Iterable

import mlflow
import numpy as np
from numpy import ndarray
from torch.utils.data import DataLoader, Sampler

from data_types.AgenticImage import ImageData
from dataset.COCODataset import COCODataset
from dataset.Utils import Utils
from dataset.agentic_coco_image import CocoImageData
from dataset.enhanced_coco import AnnotationFileAndImagePath
from dataset.samplers import IndexSampler
from juror_client import JurorClient
from juror_shared.models_v1 import ScoringResponsePayloadV1
from training import mlflow_helper
from training.degrading.degrading_functions import AbstractImageDegradingFunction
from training.hyperparameter_registry import HyperparameterRegistry
from training.mlflow_utils import mlflow_logging
from training.preprocessing.abstract_preprocessor import AbstractPreprocessor
from training.rl_training.training_params import TransformPreprocessingParams, GeneralPreprocessingParams, \
    TrainingExecutionParams
from training.split_ratios import SplitRatios
from transformer.AbstractTransformer import AbstractTransformer
from utils.CocoBuilder import CocoBuilder
from utils.ImageUtils import ImageUtils
from utils.Registries import TRANSFORMER_REGISTRY

logger = logging.getLogger(__name__)


class DegradeAndSplitPreprocessorResult:
    def __init__(self, effective_destination_root_path: Path, processed_splits: Dict[str, AnnotationFileAndImagePath]):
        self.effective_destination_root_path = effective_destination_root_path
        self.processed_splits: Dict[str, AnnotationFileAndImagePath] = processed_splits


class DegradeAndSplitPreprocessor(AbstractPreprocessor[DegradeAndSplitPreprocessorResult]):
    """Preprocessor-Klasse, die Bilder degradiert und das Dataset in Splits aufteilt.

    Diese Klasse übernimmt:
    - Anwenden einer `AbstractImageDegradingFunction` auf Bilder des Quell-Datasets.
    - Erzeugen von Splits (z. B. train/val/test) entsprechend `SplitRatios`.
    - Speichern der degradierten Bilder in jeweils eigenen Verzeichnissen.
    - Erzeugen und Speichern von COCO-Annotationen via `CocoBuilder`.
    - Bewerten der degradierten Bilder durch den `JurorClient` und Hinzufügen von Score-Annotationen.
    """

    def __init__(self, preprocessing_step: Optional[int], juror_client: Optional[JurorClient] = None):
        super().__init__(preprocessing_step)
        transform_params = HyperparameterRegistry.get_store(TransformPreprocessingParams).get()
        training_params = HyperparameterRegistry.get_store(TrainingExecutionParams).get()
        general_params = HyperparameterRegistry.get_store(GeneralPreprocessingParams).get()
        self.degradation_function: Optional[AbstractImageDegradingFunction] = None
        self.split_ratios: SplitRatios = transform_params["split"]
        self.batch_size = transform_params["batch_size"]
        self.source_path: Optional[Path] = None
        self.source_annotation_file: Optional[Path] = None
        self.image_volume_path = self.get_image_root_path()
        self.destination_root_path: Optional[Path] = None
        self.random_generator = random.Random(general_params["random_seed"])
        self.juror_client = juror_client if juror_client is not None else JurorClient(
            use_local=training_params["use_local_juror"])
        self.processed_splits: Dict[str, AnnotationFileAndImagePath] = {}
        self.debug_degradation_scoring = False  # Wenn true, wird weitertest Scoring ausgeführt um die Degradation zu validieren

    def with_source_path(self, source_path: Path) -> 'DegradeAndSplitPreprocessor':
        self.source_path = source_path
        return self

    def with_destination_root_path(self, destination_root_path: Path) -> 'DegradeAndSplitPreprocessor':
        self.destination_root_path = destination_root_path
        return self

    def with_source_annotation_file(self, source_annotation_file: Path) -> 'DegradeAndSplitPreprocessor':
        self.source_annotation_file = source_annotation_file
        return self

    def with_degradation_function(self,
                                  degradation_function: AbstractImageDegradingFunction) -> 'DegradeAndSplitPreprocessor':
        self.degradation_function = degradation_function
        return self

    @mlflow_logging("image_degradation_duration_seconds")
    def _preprocess_impl(self):
        self.effective_destination_root_path = self.image_volume_path / "degradeandsplit" / mlflow.active_run().info.experiment_id / mlflow.active_run().info.run_id / self.destination_root_path
        source_dataset: COCODataset = COCODataset(self.source_path, self.source_annotation_file)
        splits = self._create_splits(source_dataset)

        for split_name, split_indices in splits.items():
            logger.info(f"Generating {split_name} split with {len(split_indices)} images")

            # Create split-specific directory and COCO builder
            split_dir = self.effective_destination_root_path / split_name
            split_images_dir = split_dir / "images"
            split_coco_builder = CocoBuilder(source_path=self.effective_destination_root_path)
            split_coco_builder.set_description(
                f"Coco file for {split_name} split from source path {self.effective_destination_root_path} with degradation function {self.degradation_function}"
            )
            index_sampler = IndexSampler(split_indices)

            # Transform images for this split
            self._transform_images(split_coco_builder, source_dataset, index_sampler, split_images_dir)

            # Save split annotations
            split_coco_file = split_dir / "annotations.json"
            split_coco_builder.save(str(split_coco_file))
            mlflow_helper.log_artifact(local_path=str(split_coco_file), step=self.step_arg,
                                       artifact_path=str(split_name))

            mlflow_helper.log_metric(f"{split_name}_num_images", len(split_coco_builder.images))
            self.processed_splits[split_name] = AnnotationFileAndImagePath(split_coco_file, split_images_dir)
            logger.info("Finished split %s with %d images", split_name, len(split_indices))

        logger.info("Finished generating all splits")

    def _transform_images(self, coco_builder: CocoBuilder, source_dataset: COCODataset, sampler: Sampler,
                          target_directory_root: Path):
        """Transforms the images in the source dataset using the degradation function, updates the COCO builder and stores the images."""
        dataloader = DataLoader(source_dataset, batch_size=self.batch_size, sampler=sampler,
                                collate_fn=Utils.collate_keep_size)

        for batch_index, batch in enumerate(dataloader):
            self._process_image_batch(batch, coco_builder, target_directory_root)
            logger.info(f"Finished processing batch {batch_index} with {len(batch)} images")

    def _process_image_batch(self, batch: Iterable[CocoImageData], coco_builder: CocoBuilder,
                             split_images_dir: Path):
        """Processes single batch of images: Apply transformations, update COCO builder, store images"""
        for image_data in batch:
            degraded_image, transformer_labels = self.degradation_function.degrade(image_data.get_image_data("BGR"))
            transformed_image_path = str(split_images_dir / image_data.image_relative_path)
            ImageUtils.save_image(degraded_image, transformed_image_path)
            image_id = coco_builder.add_image(file_name=str(image_data.image_relative_path),
                                              width=image_data.width,
                                              height=image_data.height,
                                              initial_score=image_data.initial_score)
            scoring_response: ScoringResponsePayloadV1 = self.juror_client.score_ndarray_bgr(degraded_image)
            coco_builder.add_image_score_annotation(image_id=image_id, score=scoring_response.score,
                                                    initial_score=image_data.initial_score)
            if self.debug_degradation_scoring:
                self._debug_degradation_scoring(degraded_image,
                                                image_data,
                                                image_id,
                                                transformed_image_path,
                                                transformer_labels)

            for transformer_label in transformer_labels:
                coco_builder.add_image_transformation_annotation(image_id, transformer_name=transformer_label)

    def _debug_degradation_scoring(self, degraded_image: ndarray, image_data: CocoImageData, image_id: int,
                                   transformed_image_path: str, transformer_labels: list[str]):
        # Rollback transformation to check initial score
        degrading_transformer: AbstractTransformer = TRANSFORMER_REGISTRY.get(transformer_labels[0])
        reversing_transformer_label = degrading_transformer.get_reverse_transformer_label()
        if reversing_transformer_label:
            initial_image_for_rescoring = image_data.get_image_data("BGR")
            rescore_initial_scoring_response = self.juror_client.score_ndarray_bgr(
                initial_image_for_rescoring).score
            if abs(rescore_initial_scoring_response - image_data.initial_score) > 1e-5:
                logger.warning(f"Initial score mismatch for image {image_data.image_relative_path}: "
                               f"stored={image_data.initial_score}, live={rescore_initial_scoring_response}")
            reverting_transformer: AbstractTransformer = TRANSFORMER_REGISTRY.get(reversing_transformer_label)
            dedegraded_image = reverting_transformer.transform(degraded_image)
            ImageUtils.save_image(image_data.get_image_data("BGR"),
                                  transformed_image_path.replace(".png", "_initial.png"))
            ImageUtils.save_image(dedegraded_image, transformed_image_path.replace(".png", "_dedegraded.png"))
            dedegraded_scoring_response = self.juror_client.score_ndarray_bgr(dedegraded_image).score

            logger.warning("%s: Initial: %.5f, Rescored: %.5f, Dedegraded: %.5f for image %s",
                           image_data.image_relative_path, image_data.initial_score,
                           rescore_initial_scoring_response, dedegraded_scoring_response, image_id)

            if abs(dedegraded_scoring_response - image_data.initial_score) > 1e-5:
                logger.warning(f"Rescored initial score mismatch for image {image_data.image_relative_path}: "
                               f"stored={image_data.initial_score}, live={dedegraded_scoring_response}")

            if abs(dedegraded_scoring_response - rescore_initial_scoring_response) > 1e-5:
                logger.warning(f"Rescored initial score mismatch for image {image_data.image_relative_path}: "
                               f"initial_rescored={rescore_initial_scoring_response}, dedegraded={dedegraded_scoring_response}")

            if not np.array_equal(initial_image_for_rescoring, dedegraded_image):
                logger.warning(
                    f"Dedegraded image does not match the original for image {image_data.image_relative_path}")
        else:
            logger.warning(f"No reversable transformer for {degrading_transformer.label}")

    def _create_splits(self, source_dataset: COCODataset) -> Dict[str, List[int]]:
        total_images = len(source_dataset)
        indices = list(range(total_images))
        self.random_generator.shuffle(indices)
        return self.split_ratios.get_split_indices(indices)

    def _create_subset_dataset(self, source_dataset: COCODataset, indices: list) -> list[ImageData]:
        """Create a subset of the dataset based on indices."""
        return [source_dataset[i] for i in indices]

    def get_preprocessing_result(self) -> DegradeAndSplitPreprocessorResult:
        return DegradeAndSplitPreprocessorResult(self.effective_destination_root_path, self.processed_splits)
