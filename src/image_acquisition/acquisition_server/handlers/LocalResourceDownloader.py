import logging
from pathlib import Path

from image_acquisition.acquisition_server.handlers.AbstractHandler import AbstractHandler
from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)


class LocalResourceDownloader(AbstractHandler):
    def __init__(self, dataset_id: str, resource_file_path: str, provided_destination_path: Path, archive_root: str, target_hash: str = None):
        super().__init__(dataset_id, provided_destination_path, target_hash)
        self.archive_root = Path(archive_root)

        # Falls relativ: relativ zum Verzeichnis dieser Datei (`__file__`) auflösen
        rp = Path(resource_file_path)
        if not rp.is_absolute():
            rp = (Path(__file__).parent / 'resources' / rp).resolve()
        self.resource_file_path = rp
        logger.info(f"Downloading as local ressource {rp}")

    def _process_impl(self):
        logger.info("Copying %s", self.resource_file_path)
        downloaded_file = ImageAcquisitionUtils.copy_resource_file(self.resource_file_path, self.temp_dir, None, True)
        logger.debug("Copy completed %s", self.resource_file_path)
        logger.info(f"Extracting {downloaded_file}")
        ImageAcquisitionUtils.extract_tar(downloaded_file, self.destination_path, self.archive_root)
        logger.debug(f"Extraction of {downloaded_file} completed")
        logger.debug(f"Deleting {downloaded_file}")
        ImageAcquisitionUtils.cleanup_temp_file(downloaded_file)
        logger.info("Copied and extracted %s", self.resource_file_path)
