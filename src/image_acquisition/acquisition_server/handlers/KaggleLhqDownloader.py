import logging
import os
import shutil
import zipfile
from pathlib import Path

from image_acquisition.acquisition_server.handlers.KaggleDownloader import KaggleDownloader
from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class KaggleLhqDownloader(KaggleDownloader):
    """Handler to download LHQ dataset from Kaggle and extract only mountain images.

    This handler extends KaggleDownloader to filter and extract only images
    from a specific category (e.g., mountain) instead of extracting all files.

    Requires:
        - kaggle package installed (pip install kaggle)

    Attributes:
        dataset_id (str): Unique identifier for the dataset
        kaggle_dataset (str): Kaggle dataset identifier (e.g., 'arnaud58/landscape-pictures')
        provided_destination_path (Path): Destination path for extracted files
        category (str): Category to filter (default: 'mountain')
        target_hash (str, optional): Expected hash value for verification
    """

    def __init__(self, dataset_id: str, kaggle_dataset: str, provided_destination_path: Path,
                 category: str = "mountain", target_hash: str = None):
        # Pass None for archive_root since we handle extraction differently
        super().__init__(dataset_id, kaggle_dataset, provided_destination_path, None, target_hash)
        self.category = category.lower()

    def _extract_category_images(self, zip_file: Path) -> None:
        """Extract only images from a specific category.

        Args:
            zip_file: Path to the downloaded zip file
        """
        os.makedirs(self.destination_path, exist_ok=True)
        abs_dest = os.path.abspath(self.destination_path)

        extracted_count = 0
        skipped_count = 0

        logger.info(f"Extracting {self.category} images from {zip_file} to {abs_dest}")

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                all_files = zip_ref.namelist()

                # Filter for category images
                category_files = []
                available_categories = set()

                for file in all_files:
                    if file.endswith('/'):
                        continue

                    # Track available categories
                    parts = file.split('/')
                    if len(parts) > 1:
                        available_categories.add(parts[0].lower())

                    # Check if file is in desired category and is an image
                    if (self.category in file.lower() and
                        file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))):
                        category_files.append(file)

                if not category_files:
                    logger.warning(f"No files found with category '{self.category}'")
                    logger.info(f"Available categories: {sorted(available_categories)}")
                    raise ValueError(f"No {self.category} images found in dataset. Available categories: {sorted(available_categories)}")

                logger.info(f"Found {len(category_files)} {self.category} images to extract")

                # Extract filtered files
                for file in category_files:
                    try:
                        # Skip absolute paths
                        if os.path.isabs(file):
                            logger.warning(f"Skipping absolute path: {file}")
                            skipped_count += 1
                            continue

                        # Calculate destination path (flatten structure, use only filename)
                        dest_path = os.path.join(abs_dest, os.path.basename(file))

                        # Path traversal check
                        if os.path.commonpath([abs_dest, os.path.abspath(dest_path)]) != abs_dest:
                            logger.warning(f"Skipping path traversal entry: {file}")
                            skipped_count += 1
                            continue

                        # Extract file
                        with zip_ref.open(file, "r") as src, open(dest_path, "wb") as out_f:
                            shutil.copyfileobj(src, out_f)

                        extracted_count += 1
                        if extracted_count % 100 == 0:
                            logger.info(f"Extracted {extracted_count} {self.category} images...")

                    except Exception as e:
                        logger.warning(f"Could not extract {file}: {e}")
                        skipped_count += 1
                        continue

                logger.info(f"Extraction completed: {extracted_count} images extracted, {skipped_count} skipped")

        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {zip_file}")
            raise RuntimeError(f"Unable to read zip file {zip_file}: {e}") from e

    def _process_impl(self):
        """Main processing logic: download from Kaggle and extract category-filtered images."""
        try:
            # Download from Kaggle (using parent class method)
            zip_file = self._download_from_kaggle()

            # Extract only category-specific images (override parent behavior)
            self._extract_category_images(zip_file)

            # Cleanup downloaded zip
            logger.info(f"Cleaning up temporary file: {zip_file}")
            ImageAcquisitionUtils.cleanup_temp_file(zip_file)

            # Remove temp directory if empty
            try:
                if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                    shutil.rmtree(self.temp_dir)
                    logger.info(f"Removed empty temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Could not remove temp directory {self.temp_dir}: {e}")

            logger.info(f"Kaggle {self.category} download completed for {self.kaggle_dataset}")

        except Exception as e:
            logger.exception(f"Error processing Kaggle {self.category} download: {e}")
            raise
