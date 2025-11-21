import logging
from collections.abc import Callable
from pathlib import Path

from image_acquisition.acquisition_server.handlers.AbstractHandler import AbstractHandler
from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ArchiveDownloader(AbstractHandler):
    """Handler zum Herunterladen und Extrahieren von Archivdateien (z.B. ZIP, TAR).
    Unterstützt verschiedene Extraktionsfunktionen, die beim Initialisieren übergeben werden können.
    Attributes:
        dataset_id (str): Eindeutige Kennung des Datensatzes.
        url (str): URL der Archivdatei zum Herunterladen.
        provided_destination_path (Path): Zielpfad für die extrahierten Dateien.
        archive_root (str): Wurzelverzeichnis innerhalb des Archivs, das extrahiert werden soll.
        extractor_fn (Callable): Funktion zum Extrahieren der Archivdatei.
        target_hash (str, optional): Erwarteter Hash-Wert des extrahierten Inhalts zur Verifikation.
    """
    def __init__(self, dataset_id: str, url: str, provided_destination_path: Path, archive_root: str,
                 extractor_fn: Callable, target_hash: str = None):
        super().__init__(dataset_id, provided_destination_path, target_hash)
        self.archive_root = Path(archive_root)
        self.extractor_fn = extractor_fn
        self.url = url

    def _process_impl(self):
        logger.info("Downloading %s", self.url)
        downloaded_file = ImageAcquisitionUtils.download_file(self.url, self.temp_dir)
        logger.debug("Download completed %s", self.url)
        logger.info("Extracting file %s", downloaded_file)
        self.extractor_fn(downloaded_file, self.destination_path, self.archive_root)
        logger.debug("Extraction completed for %s", downloaded_file)
        logger.debug("Deleting %s", downloaded_file)
        ImageAcquisitionUtils.cleanup_temp_file(downloaded_file)
        logger.info("Downloading and extraction finished for %s", self.url)
