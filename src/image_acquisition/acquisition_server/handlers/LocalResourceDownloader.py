import logging
import os
from pathlib import Path

from image_acquisition.acquisition_server.handlers.AbstractHandler import AbstractHandler
from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LocalResourceDownloader(AbstractHandler):
    def __init__(self, resource_file_path: str, destination_path: str, target_hash: str = None):
        super().__init__(destination_path, target_hash)


        # Falls relativ: relativ zum Verzeichnis dieser Datei (`__file__`) auflösen
        rp = Path(resource_file_path)
        if not rp.is_absolute():
            rp = (Path(__file__).parent / 'resources' / rp).resolve()
        self.resource_file_path = str(rp)

        print(f"Initialized LocalResourceDownloader with resource_file_path: {self.resource_file_path}")

        self.temp_dir = Path(self.destination_path, "./temp")


    def _process_impl(self):
        logger.info("Copying %s", self.resource_file_path)
        downloaded_file = ImageAcquisitionUtils.copy_resource_file(self.resource_file_path, str(self.temp_dir))
        logger.debug("Copy completed %s", self.resource_file_path)
        logger.info(f"Extracting {downloaded_file}")
        ImageAcquisitionUtils.extract_tar(downloaded_file, self.destination_path)
        logger.debug(f"Extraction of {downloaded_file} completed")
        logger.debug(f"Deleting {downloaded_file}")
        ImageAcquisitionUtils.cleanup_temp_file(downloaded_file)
        logger.info("Copied and extracted %s", self.resource_file_path)
