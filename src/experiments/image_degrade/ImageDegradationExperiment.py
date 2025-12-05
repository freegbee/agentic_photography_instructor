import logging
import os
import random
import time
from pathlib import Path
from typing import Optional, Dict, Iterable, cast, Tuple

from numpy import ndarray
from torch.utils.data import Dataset, DataLoader

from data_types.AgenticImage import ImageData
from data_types.ImageDatasetConfiguration import ImageDatasetConfiguration
from dataset.COCODataset import COCODataset
from experiments.image_degrade.BatchImageDegradationMetricAccumulator import BatchImageDegradationMetricAccumulator
from experiments.shared.PhotographyExperiment import PhotographyExperiment
from experiments.shared.Utils import Utils
from juror_client import JurorClient
from juror_shared.models_v1 import ScoringResponsePayloadV1
from transformer import REVERSIBLE_TRANSFORMERS
from transformer.AbstractTransformer import AbstractTransformer
from transformer.color_adjustment import InvertColorChannelTransformerGR
from utils.CocoBuilder import CocoBuilder
from utils.ConfigLoader import ConfigLoader
from utils.ImageUtils import ImageUtils
from utils.Registries import TRANSFORMER_REGISTRY

logger = logging.getLogger(__name__)

class ImageDegradationExperiment(PhotographyExperiment):
    def __init__(self, experiment_name: str, target_directory_root: str, target_dataset_id: str, transformer_name: str, run_name: str = None, source_dataset_id: str = "single_image", batch_size: int = 2, split_ratios: Tuple[float, float, float] = None, seed: int = 42):
        super().__init__(experiment_name)
        # Zeile nur um Import-Fehler zu vermeiden :-(
        abstract_transformer: AbstractTransformer = InvertColorChannelTransformerGR()

        logger.info("Experiment name: %s with target_directory_root %s and source_dataset_id %s", experiment_name, target_directory_root, source_dataset_id)
        self.target_directory_root = Path(os.environ["IMAGE_VOLUME_PATH"]) / target_directory_root
        self.target_dataset_id = target_dataset_id
        self.use_random_transformer = transformer_name == "RANDOM"

        if self.use_random_transformer:
            self.transformer_choice = REVERSIBLE_TRANSFORMERS
        else:
            self.transformer: AbstractTransformer = TRANSFORMER_REGISTRY.get(transformer_name)
        self.run_name = run_name
        self.source_dataset_id = source_dataset_id
        self.batch_size = batch_size
        self.jurorClient = None

        # Split configuration for train/val/test
        self.split_ratios = split_ratios  # e.g., (0.7, 0.15, 0.15) for train/val/test
        self.seed = seed
        self.random_generator = random.Random(seed)


    def configure(self, config: dict):
        pass

    def _get_tags_for_run(self):
        return {"dataset_id": self.source_dataset_id}

    def _get_run_name(self) -> Optional[str]:
        return self.run_name

    def _run_impl(self, experiment_created, active_run):
        # TODO Die ganze Vorbereitung dürfte sich mit ImageScoringPhotographyExperiment teilen lassen
        #      Superklasse erweitern? Utility-Klasse?
        try:
            config_dict: Dict = ConfigLoader.load_dataset_config(self.source_dataset_id)
        except Exception as e:
            logger.exception("Exception loading config for dataset %s: %s", self.source_dataset_id, e)
            raise e
        # Dataset-Konfiguration laden und Bildpfad ermitteln
        self.dataset_config = ImageDatasetConfiguration.from_dict(self.source_dataset_id, config_dict)

        start_ensure_images = time.perf_counter()
        image_dataset_hash = Utils.ensure_image_dataset(self.dataset_config.dataset_id)
        self.log_metric("ensure_image_dataset_duration_seconds", time.perf_counter() - start_ensure_images)
        self.log_param("dataset_hash", image_dataset_hash)

        source_images_root_path = self.dataset_config.calculate_images_root_path()
        logger.debug("Image root path calculated as %s", source_images_root_path)

        self.log_param("experiment_type", "dataset image degradation")
        self.log_param("source_dataset_id", self.dataset_config.dataset_id)
        self.log_param("source_images_root_path", source_images_root_path)
        self.log_param("target_dataset_id", self.target_dataset_id)
        self.log_param("target_images_directory_root", self.target_directory_root)
        self.log_param("batch_size", self.batch_size)
        self.log_param("transformer", self.transformer.label if not self.use_random_transformer else "RANDOM")

        # COCO-Builder initialisieren
        self.coco_builder = CocoBuilder(self.source_dataset_id)
        self.coco_builder.set_description(f"Coco file for dataset {self.source_dataset_id} and transformer {self.transformer.label if not self.use_random_transformer else 'RANDOM'} for image degradation experiment")

        source_dataset = COCODataset(source_images_root_path, self.dataset_config.calculate_annotations_file_path())

        # JurorClient initialisieren
        self.jurorClient = JurorClient(os.environ["JUROR_SERVICE_URL"])

        # Generate splits if configured
        if self.split_ratios is not None:
            self._generate_splits(source_dataset)
        else:
            self._transform_images(self.coco_builder, source_dataset, self.target_directory_root)

            # Nachverarbeitung
            self.log_metric("total_number_of_images", len(self.coco_builder.images))

            coco_file_path = self.target_directory_root / "annotations.json"
            self.coco_builder.save(str(coco_file_path))

            self.log_artifact(local_path=str(coco_file_path))

        # Ergebnisse loggen
        logger.debug(f"Image transformation complete for dataset %s", self.source_dataset_id)
        logger.debug(f"Coco %s", self.coco_builder.to_json_string())

        logger.info("Finished Image Transformation Experiment")


    def _transform_images(self, coco_builder: CocoBuilder, source_dataset, target_directory_root: Path):
        """Transformiert die Bilder im Dataset und speichert sie im Zielverzeichnis ab."""
        # Handle both COCODataset and list of ImageData
        if isinstance(source_dataset, list):
            # For list, create batches manually
            metrics_accumulator = BatchImageDegradationMetricAccumulator()
            for batch_index in range(0, len(source_dataset), self.batch_size):
                batch = source_dataset[batch_index:batch_index + self.batch_size]
                metrics_accumulator.reset()
                self._process_image_batch(batch_index, batch, coco_builder, target_directory_root, metrics_accumulator)
                self.log_batch_metrics(metrics_accumulator.compute_metrics(), batch_index)
        else:
            # For COCODataset, use DataLoader
            dataloader = DataLoader(cast(Dataset[ImageData], source_dataset), batch_size=self.batch_size, collate_fn=Utils.collate_keep_size, num_workers=4)
            metrics_accumulator = BatchImageDegradationMetricAccumulator()
            for batch_index, batch in enumerate(dataloader):
                metrics_accumulator.reset()
                self._process_image_batch(batch_index, batch, coco_builder, target_directory_root, metrics_accumulator)
                self.log_batch_metrics(metrics_accumulator.compute_metrics(), batch_index)

    def _process_image_batch(self, batch_index: int, batch: Iterable[ImageData], coco_builder: CocoBuilder, target_directory_root: Path, metrics_accumulator: BatchImageDegradationMetricAccumulator):
        """Verarbeitet einen Batch von Bildern: wendet die Transformation an, speichert die Bilder und aktualisiert den COCO-Builder."""
        metrics_accumulator.start(batch_index)
        for image_data in batch:
            degraded_image, transformer_label = self._apply_transformer(image_data)
            transformed_image_path = str(Path(target_directory_root / "images" / image_data.image_relative_path))
            logger.debug("Writing image to %s", transformed_image_path)
            ImageUtils.save_image(degraded_image, transformed_image_path)
            image_id = coco_builder.add_image(str(image_data.image_relative_path), 0, 0)
            logger.info("Transformed image %s with transformer %s", image_data.image_path, transformer_label)
            scoring_response: ScoringResponsePayloadV1 = self.jurorClient.score_image(transformed_image_path)
            logger.info("scoring response %s", scoring_response)
            coco_builder.add_image_transformation_score_annotation(image_id, scoring_response.score, image_data.score, transformer_name=transformer_label)
            coco_builder.add_image_final_score_annotation(image_id, scoring_response.score, image_data.score)
            metrics_accumulator.add_score(scoring_response.score, image_data.score)

        metrics_accumulator.stop()

    def _apply_transformer(self, image_data: ImageData) -> Tuple[ndarray, str]:
        t = None
        if self.use_random_transformer:
            transformer_label = self.random_generator.choice(self.transformer_choice)
            t = TRANSFORMER_REGISTRY.get(transformer_label)
        else:
            t = self.transformer

        return t.transform(image_data.get_image_data("BGR")), t.label

    def _generate_splits(self, source_dataset: COCODataset):
        """Generate train/val/test splits from the source dataset."""
        train_ratio, val_ratio, test_ratio = self.split_ratios
        total_images = len(source_dataset)

        # Shuffle indices
        indices = list(range(total_images))
        self.random_generator.shuffle(indices)

        # Calculate split sizes
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        splits = [
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices)
        ]

        self.log_param("split_ratios", f"{train_ratio}/{val_ratio}/{test_ratio}")
        self.log_param("train_size", len(train_indices))
        self.log_param("val_size", len(val_indices))
        self.log_param("test_size", len(test_indices))

        for split_name, split_indices in splits:
            logger.info(f"Generating {split_name} split with {len(split_indices)} images")

            # Create split-specific directory and COCO builder
            split_dir = self.target_directory_root / split_name
            split_coco_builder = CocoBuilder(f"{self.source_dataset_id}_{split_name}")
            split_coco_builder.set_description(
                f"Coco file for {split_name} split of dataset {self.source_dataset_id} with transformer {self.transformer.label if not self.use_random_transformer else 'RANDOM'}"
            )

            # Create subset dataset
            split_dataset = self._create_subset_dataset(source_dataset, split_indices)

            # Transform images for this split
            self._transform_images(split_coco_builder, split_dataset, split_dir)

            # Save split annotations
            split_coco_file = split_dir / "annotations.json"
            split_coco_builder.save(str(split_coco_file))
            self.log_artifact(local_path=str(split_coco_file))

            self.log_metric(f"{split_name}_num_images", len(split_coco_builder.images))

        logger.info("Finished generating all splits")

    def _create_subset_dataset(self, source_dataset: COCODataset, indices: list) -> list:
        """Create a subset of the dataset based on indices."""
        return [source_dataset[i] for i in indices]
