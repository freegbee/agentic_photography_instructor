import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AbstractHandler(ABC):

    def __init__(self, destination_path, target_hash: str = None):
        provided_dest = Path(destination_path)
        image_volume = os.getenv("IMAGE_VOLUME_PATH")

        if provided_dest.is_absolute():
            final_dest = provided_dest
        elif image_volume:
            final_dest = (Path(image_volume) / provided_dest)
        else:
            final_dest = provided_dest

        # Auf absoluten, aufgelösten Pfad normalisieren
        self.destination_path = str(final_dest.resolve())
        self.target_hash = target_hash


    @abstractmethod
    def _process_impl(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _check_hash(self, path: str, target_hash: str) -> bool | None:
        try:
            logger.debug(f"Checking hash for path %s", path)
            if self.target_hash is None:
                return True
            computed_hash = ImageAcquisitionUtils.compute_dir_hash(path)
            logger.info("Computed hash %s and target hash %s: Matching=%s", computed_hash, target_hash, computed_hash == target_hash)
            return computed_hash == self.target_hash
        except Exception as e:
            logger.exception("Exception checking hash for path %s: %s", path, e)
            print(f"Error computing hash for {path}: {e}")


    def process(self) -> str | None:
        if self._check_hash(self.destination_path, self.target_hash):
            return self.target_hash

        self._process_impl()
        new_hash = ImageAcquisitionUtils.compute_dir_hash(self.destination_path)
        logger.info("Computed hash after processing: %s", new_hash)
        return new_hash

