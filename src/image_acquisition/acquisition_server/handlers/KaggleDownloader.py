import logging
import os
import shutil
from pathlib import Path

from image_acquisition.acquisition_server.handlers.AbstractHandler import AbstractHandler
from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class KaggleDownloader(AbstractHandler):
    """Handler to download datasets from Kaggle and extract them.

    This is a generic handler for downloading any Kaggle dataset and extracting
    the specified root directory from the archive, similar to ZipDownloader
    but specifically for Kaggle datasets.

    Requires:
        - kaggle package installed (pip install kaggle)

    Attributes:
        dataset_id (str): Unique identifier for the dataset
        kaggle_dataset (str): Kaggle dataset identifier (e.g., 'username/dataset-name')
        provided_destination_path (Path): Destination path for extracted files
        archive_root (str): Root directory within the archive to extract
        target_hash (str, optional): Expected hash value for verification
    """

    def __init__(self, dataset_id: str, kaggle_dataset: str, provided_destination_path: Path,
                 archive_root: str, target_hash: str = None):
        super().__init__(dataset_id, provided_destination_path, target_hash)
        self.kaggle_dataset = kaggle_dataset
        self.archive_root = Path(archive_root) if archive_root else None

    def _download_from_kaggle(self) -> Path:
        """Download dataset from Kaggle using Kaggle API.

        Returns:
            Path to the downloaded zip file

        Raises:
            ImportError: If kaggle package is not installed
            Exception: If Kaggle API authentication fails or download fails
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            logger.error("kaggle package not installed. Install with: pip install kaggle")
            raise ImportError("kaggle package required. Install with: pip install kaggle")

        logger.info("Authenticating with Kaggle API...")
        try:
            api = KaggleApi()
            api.authenticate()

        except Exception as e:
            logger.error("Kaggle API authentication failed.")
            raise Exception(f"Kaggle authentication failed: {e}") from e

        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)

        logger.info(f"Downloading Kaggle dataset: {self.kaggle_dataset} to {self.temp_dir}")
        try:
            api.dataset_download_files(self.kaggle_dataset, path=str(self.temp_dir), unzip=False)
        except Exception as e:
            logger.error(f"Failed to download dataset {self.kaggle_dataset}: {e}")
            raise Exception(f"Kaggle download failed: {e}") from e

        # Find the downloaded zip file
        zip_files = list(self.temp_dir.glob('*.zip'))
        if not zip_files:
            raise FileNotFoundError(f"No zip file found after Kaggle download in {self.temp_dir}")

        logger.info(f"Downloaded zip file: {zip_files[0]}")
        return zip_files[0]

    def _log_zip_structure(self, zip_file: Path):
        """Log the structure of the zip file for debugging."""
        import zipfile

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                logger.info(f"Zip file contains {len(all_files)} entries")

                # Log first 10 entries to understand structure
                sample_files = all_files[:10]
                logger.info(f"Sample entries from zip: {sample_files}")

                # Identify top-level directories
                top_level_dirs = set()
                for f in all_files:
                    parts = f.split('/')
                    if len(parts) > 1:
                        top_level_dirs.add(parts[0])

                logger.info(f"Top-level directories in zip: {sorted(top_level_dirs)}")
        except Exception as e:
            logger.warning(f"Could not inspect zip structure: {e}")

    def _process_impl(self):
        """Main processing logic: download from Kaggle and extract."""
        try:
            # Download from Kaggle
            zip_file = self._download_from_kaggle()

            # Log zip structure for debugging
            self._log_zip_structure(zip_file)

            # Extract using the standard zip extraction utility
            logger.info(f"Extracting {zip_file} to {self.destination_path} with archive_root={self.archive_root}")
            ImageAcquisitionUtils.extract_zip(zip_file, self.destination_path, self.archive_root)

            # Cleanup downloaded zip
            # logger.info(f"Cleaning up temporary file: {zip_file}")
            # ImageAcquisitionUtils.cleanup_temp_file(zip_file)

            # Remove temp directory if empty
            try:
                if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                    shutil.rmtree(self.temp_dir)
                    logger.info(f"Removed empty temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Could not remove temp directory {self.temp_dir}: {e}")

            logger.info(f"Kaggle download completed for {self.kaggle_dataset}")

        except Exception as e:
            logger.exception(f"Error processing Kaggle download: {e}")
            raise
