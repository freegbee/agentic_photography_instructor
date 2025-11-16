import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AbstractHandler(ABC):

    def __init__(self, dataset_id: str, provided_destination_path: Path, target_hash: str = None):
        self.dataset_id: str = dataset_id
        self.temp_dir: Path = self.calculate_temp_dir()
        self.destination_path: Path = self.calculate_destination_path(provided_destination_path)
        self.target_hash: str = target_hash

    @staticmethod
    def get_image_volume_path() -> Path:
        image_volume = os.getenv("IMAGE_VOLUME_PATH")
        if not image_volume:
            raise ValueError("IMAGE_VOLUME_PATH environment variable is not set")
        return Path(image_volume)

    def calculate_temp_dir(self) -> Path:
        return AbstractHandler.get_image_volume_path() / "temp" / self.dataset_id

    @staticmethod
    def calculate_destination_path(provided_destination_path) -> Path:
        provided_dest = Path(provided_destination_path)
        if provided_dest.is_absolute():
            final_dest = provided_dest
        else:
            final_dest = AbstractHandler.get_image_volume_path() / provided_dest
        return final_dest.resolve()

    @abstractmethod
    def _process_impl(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _check_hash(self, path: Path, target_hash: str) -> bool | None:
        try:
            logger.debug(f"Checking hash for path %s", path)
            if self.target_hash is None:
                return True
            computed_hash = ImageAcquisitionUtils.compute_dir_hash(path)
            logger.info("Computed hash %s and target hash %s: Matching=%s", computed_hash, target_hash,
                        computed_hash == target_hash)
            return computed_hash == self.target_hash
        except Exception as e:
            logger.exception("Exception checking hash for path %s: %s", path, e)
            print(f"Error computing hash for {path}: {e}")

    def process(self) -> str | None:
        if self._check_hash(self.destination_path, self.target_hash):
            logger.info("Target hash %s matches existing data at %s, skipping processing", self.target_hash, self.destination_path)
            return self.target_hash

        self._process_impl()
        new_hash = ImageAcquisitionUtils.compute_dir_hash(self.destination_path)
        logger.info("Computed hash after processing: %s", new_hash)
        return new_hash
