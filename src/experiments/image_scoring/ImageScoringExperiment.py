import logging
import os
import time
from typing import Dict, Optional

from torch.utils.data import Dataset, DataLoader

from data_types.ImageDatasetConfiguration import ImageDatasetConfiguration
from experiments.image_scoring.BatchImageScoringMetricAccumulator import BatchImageScoringMetricAccumulator
from experiments.shared.PhotographyExperiment import PhotographyExperiment
from experiments.shared.Utils import Utils
from image_acquisition.AbsolutePathImageDataset import AbsolutePathImageDataset
from juror_client import JurorClient
from juror_shared.models_v1 import ScoringResponsePayloadV1
from utils.CocoBuilder import CocoBuilder
from utils.ConfigLoader import ConfigLoader

logger = logging.getLogger(__name__)


class ImageScoringPhotographyExperiment(PhotographyExperiment):

    def __init__(self, experiment_name: str, run_name: str = None, dataset_id: str = "single_image",
                 batch_size: int = 2):
        super(ImageScoringPhotographyExperiment, self).__init__(experiment_name)
        self.run_name = run_name
        self.dataset_id = dataset_id
        self.batch_size = batch_size
        self.coco_builder = None
        self.dataset_config = None
        self.jurorClient = None

    def configure(self, config: dict):
        pass

    def _get_tags_for_run(self):
        return {"dataset_id": self.dataset_id}

    def _get_run_name(self) -> Optional[str]:
        return self.run_name

    def _run_impl(self, experiment_created, active_run):
        try:
            config_dict: Dict = ConfigLoader.load_dataset_config(self.dataset_id)
        except Exception as e:
            logger.exception("Exception loading config for dataset %s: %s", self.dataset_id, e)
            raise e
        # Dataset-Konfiguration laden und Bildpfad ermitteln
        self.dataset_config = ImageDatasetConfiguration.from_dict(self.dataset_id, config_dict)
        start_ensure_images = time.perf_counter()
        image_dataset_hash = Utils.ensure_image_dataset(self.dataset_config.dataset_id)
        self.log_metric("ensure_image_dataset_duration_seconds", time.perf_counter() - start_ensure_images)
        self.log_param("dataset_hash", image_dataset_hash)

        images_root_path = self.dataset_config.calculate_images_root_path()
        logger.debug("Image root path calculated as %s", images_root_path)

        self.log_param("experiment_type", "coco file creation and image scoring")
        self.log_param("dataset_id", self.dataset_config.dataset_id)
        self.log_param("images_root_path", images_root_path)
        self.log_param("batch_size", self.batch_size)

        # COCO-Builder initialisieren
        self.coco_builder = CocoBuilder(self.dataset_id)
        self.coco_builder.set_description(
            f"Coco annotations file for dataset {self.dataset_id} for image scoring experiment")

        # JurorClient initialisieren
        self.jurorClient = JurorClient(os.environ["JUROR_SERVICE_URL"])

        # Images effektiv scoren
        self._score_images(self.coco_builder, images_root_path)

        # Nachverarbeitung
        self.log_metric("total_number_of_images", len(self.coco_builder.images))

        coco_file_path = self.dataset_config.calculate_annotations_file_path()
        logger.info("Saving COCO file to %s", coco_file_path)
        self.coco_builder.save(str(coco_file_path))

        self.log_artifact(local_path=str(coco_file_path))

        # Ergebnisse loggen
        logger.debug(f"Image scoring experiment complete for dataset %s", self.dataset_id)
        logger.debug(f"Coco %s", self.coco_builder.to_json_string())

        logger.info("Finished Image Scoring Experiment")

    def _score_images(self, coco_builder, image_path: str):
        dataset: Dataset = AbsolutePathImageDataset(image_path)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Utils.collate_keep_size)
        metrics_accumulator = BatchImageScoringMetricAccumulator()
        for batch_index, batch in enumerate(dataloader):
            metrics_accumulator.reset()
            self._process_batch(batch_index, batch, coco_builder, metrics_accumulator)
            self.log_batch_metrics(metrics_accumulator.compute_metrics(), batch_index)

    def _process_batch(self, batch_index, batch, coco_builder, metrics_accumulator: BatchImageScoringMetricAccumulator):
        metrics_accumulator.start(batch_index)
        for image_full_path, path, file_name in batch:
            scoring_response: ScoringResponsePayloadV1 = self.jurorClient.score_image(image_full_path)
            image_id = coco_builder.add_image(file_name, 0, 0)
            coco_builder.add_image_score_annotation(image_id, scoring_response.score)
            metrics_accumulator.add_score(scoring_response.score)
        metrics_accumulator.stop()
