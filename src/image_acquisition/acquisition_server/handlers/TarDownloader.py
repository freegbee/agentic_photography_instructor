import logging
from pathlib import Path

from image_acquisition.acquisition_server.handlers.ArchiveDownloader import ArchiveDownloader
from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TarDownloader(ArchiveDownloader):
    def __init__(self, dataset_id: str, url: str, provided_destination_path: Path, archive_root: str,
                 target_hash: str = None):
        super().__init__(dataset_id, url, provided_destination_path, archive_root, ImageAcquisitionUtils.extract_tar,
                         target_hash)
