import logging
from abc import ABC, abstractmethod

from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AbstractHandler(ABC):

    def __init__(self, destination_path, target_hash: str = None):
        self.destination_path = destination_path
        self.target_hash = target_hash


    @abstractmethod
    def _process_impl(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _check_hash(self, path: str, target_hash: str) -> bool | None:
        try:
            print(f"Checking hash for {path}")
            logger.debug(f"Checking hash for {path}")
            if self.target_hash is None:
                return True
            computed_hash = ImageAcquisitionUtils.compute_dir_hash(path)
            print(f"Computed hash {computed_hash} and target hash {target_hash}")
            logger.info(f"Computed hash {computed_hash} and target hash {target_hash}")
            return computed_hash == self.target_hash
        except Exception as e:
            print(f"Error computing hash for {path}: {e}")


    def process(self) -> str | None:
        print("Processing - start to check hash")
        if self._check_hash(self.destination_path, self.target_hash):
            print(f"Target hash {self.target_hash} matched")
            logger.info(f"Target hash {self.target_hash} matched")
            return self.target_hash

        self._process_impl()
        new_hash = ImageAcquisitionUtils.compute_dir_hash(self.destination_path)
        print(f"Computed hash {new_hash} and target hash {self.target_hash}")
        return new_hash

