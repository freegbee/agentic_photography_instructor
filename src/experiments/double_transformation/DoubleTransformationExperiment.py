import logging
import os
import random
from pathlib import Path
from typing import Optional, Dict, Any

from data_types.ImageDatasetConfiguration import ImageDatasetConfiguration
from dataset.COCODataset import COCODataset
from dataset.TopKSampler import TopKSampler
from dataset.Utils import Utils as DatasetUtils
from experiments.shared.PhotographyExperiment import PhotographyExperiment
from experiments.shared.Utils import Utils as SharedUtils
from utils.CocoBuilder import CocoBuilder
from utils.ConfigLoader import ConfigLoader

logger = logging.getLogger(__name__)


class DoubleTransformationExperiment(PhotographyExperiment):
    def __init__(self, experiment_name: str, target_directory_root: str = "double_transformed",
                 run_name: Optional[str] = None, source_dataset_id: str = "single_image",
                 max_images: Optional[int] = None, seed: int = 42):
        super().__init__(experiment_name)
        self.run_name = run_name
        self.source_dataset_id = source_dataset_id
        self.max_images = max_images
        self.target_directory_root = Path(os.environ.get("IMAGE_VOLUME_PATH", ".")) / target_directory_root
        self.seed = seed
        self.random = random.Random(seed)
        self.coco_builder = None
        logger.info("Initialized DoubleTransformationExperiment: %s", experiment_name)

    def configure(self, config: dict):
        pass

    def _get_tags_for_run(self) -> Dict[str, Any]:
        return {"dataset_id": self.source_dataset_id}

    def _get_run_name(self) -> Optional[str]:
        return self.run_name

    def _run_impl(self, experiment_created, active_run):
        logger.info(
            "Running DoubleTransformationExperiment with source_dataset_id=%s, target_directory_root=%s, max_images=%s, seed=%d",
            self.source_dataset_id, self.target_directory_root, self.max_images, self.seed)

        try:
            config_dict = ConfigLoader.load_dataset_config(self.source_dataset_id)
        except Exception as e:
            logger.exception("Exception loading config for dataset %s: %s", self.source_dataset_id, e)
            raise

        # Dataset-Konfiguration laden und Bildpfad ermitteln
        self.dataset_config = ImageDatasetConfiguration.from_dict(self.source_dataset_id, config_dict)

        # Sicherstellen, dass Images vorhanden sind
        images_root_path = self.dataset_config.calculate_images_root_path()
        image_dataset_hash = SharedUtils.ensure_image_dataset(self.dataset_config.dataset_id)
        self.log_metric("ensure_image_dataset_duration_seconds", 0.0)
        self.log_param("dataset_hash", image_dataset_hash)

        # COCO-Input Dataset
        annotation_path = self.dataset_config.calculate_annotations_file_path()
        source_dataset = COCODataset(images_root_path, annotation_path)

        # Wenn max_images gesetzt, nutze TopKSampler -> vorhandene Utils-Funktion verwenden
        if self.max_images is not None:
            # Berechne Scores für das Dataset (nutzt dataset.Utils.calculate_dataset_scores intern)
            DatasetUtils.calculate_dataset_scores(source_dataset)
            logger.info("Calculated %d scores for dataset", len(source_dataset.scores))
            sampler = TopKSampler(source_dataset, k=int(self.max_images))
            # Note: 'sampler' is assigned here for future use in image selection, but is not yet used in this skeleton implementation.
        else:
            sampler = None

        # Output COCO-Builder initialisieren
        self.coco_builder = CocoBuilder(self.source_dataset_id)
        self.coco_builder.set_description(f"Double transformation dataset derived from {self.source_dataset_id}")

        # Ensure output directories exist
        try:
            self.target_directory_root.mkdir(parents=True, exist_ok=True)
            (self.target_directory_root / "images").mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to create target directories: %s", e)

        # Speichere initiale leere COCO Datei (wird später befüllt)
        coco_file_path = self.target_directory_root / "annotations.json"
        self.coco_builder.save(str(coco_file_path))
        # Logge das erzeugte Artefakt
        try:
            self.log_artifact(local_path=str(coco_file_path))
        except Exception:
            # logging only, nicht fatal
            logger.debug("log_artifact failed or mlflow not reachable")

        logger.info("DoubleTransformationExperiment preparation complete. Ready to process images.")

    def process_image(self, image_data, transformer1, transformer2):
        """
        Placeholder for processing a single image with two transformers.
        To be implemented in later steps.
        """
        raise NotImplementedError("process_image is not implemented in the skeleton")
