import logging
from abc import ABC
from pathlib import Path

from image_acquisition.acquisition_server.handlers.AbstractHandler import AbstractHandler
from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ZipDownloader(AbstractHandler):
    def __init__(self, url: str, destination_path: str, target_hash: str = None):
        super().__init__(destination_path, target_hash)
        self.url = url
        self.temp_dir = Path(self.destination_path, "./temp")

    def _process_impl(self):
        logger.info(f"Downloading {self.url}")
        downloaded_file = ImageAcquisitionUtils.download_file(self.url, str(self.temp_dir))
        logger.debug(f"Download completed {self.url}")
        logger.info(f"Extracting {downloaded_file}")
        ImageAcquisitionUtils.extract_zip(downloaded_file, self.destination_path)
        logger.debug(f"Extraction of {downloaded_file} completed")
        logger.debug(f"Deleting {downloaded_file}")
        ImageAcquisitionUtils.cleanup_temp_file(downloaded_file)
        logger.info(f"Downloaded and extracted {self.url}")
