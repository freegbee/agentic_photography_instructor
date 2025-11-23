import logging
import os
import random
from pathlib import Path
from typing import Optional, Dict, Any

from experiments.shared.PhotographyExperiment import PhotographyExperiment

logger = logging.getLogger(__name__)

class DoubleTransformationExperiment(PhotographyExperiment):
    def __init__(self, experiment_name: str, target_directory_root: str = "double_transformed", run_name: Optional[str] = None, source_dataset_id: str = "single_image", max_images: Optional[int] = None, seed: int = 42):
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
        # Minimal skeleton: log basic params and exit cleanly.
        logger.info("Running DoubleTransformationExperiment with source_dataset_id=%s, target_directory_root=%s, max_images=%s, seed=%d", self.source_dataset_id, self.target_directory_root, self.max_images, self.seed)
        # Ensure output directories exist
        try:
            self.target_directory_root.mkdir(parents=True, exist_ok=True)
            (self.target_directory_root / "images").mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to create target directories: %s", e)
        # No further processing in skeleton
        logger.info("DoubleTransformationExperiment skeleton finished.")

    def process_image(self, image_data, transformer1, transformer2):
        """
        Placeholder for processing a single image with two transformers.
        To be implemented in later steps.
        """
        raise NotImplementedError("process_image is not implemented in the skeleton")

