import logging
from pathlib import Path

from image_acquisition.acquisition_server.handlers.AbstractHandler import AbstractHandler
from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TarDownloader(AbstractHandler):
    def __init__(self, url: str, destination_path: str):
        self.url = url
        self.destination_path = destination_path
        self.tar_temp_dir = Path(self.destination_path, "./temp")

    def _process_impl(self):
        # TODO Nur downloaden, wenn/falls es notwendig ist. Ziel-Filesystem mit einem Hash vergleichen
        logger.info(f"Downloading {self.url}")
        downloaded_file = ImageAcquisitionUtils.download_file(self.url, str(self.tar_temp_dir))
        logger.debug(f"Download completed {self.url}")
        logger.info(f"Extracting {downloaded_file}")
        ImageAcquisitionUtils.extract_tar(downloaded_file, self.destination_path)
        logger.debug(f"Extraction of {downloaded_file} completed")
        logger.debug(f"Deleting {downloaded_file}")
        ImageAcquisitionUtils.cleanup_temp_file(downloaded_file)
        logger.info(f"Downloaded and extracted {self.url}")
