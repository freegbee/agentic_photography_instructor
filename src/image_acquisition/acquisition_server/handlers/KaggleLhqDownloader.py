import logging
import os
import re
import shutil
import zipfile
from pathlib import Path

from image_acquisition.acquisition_server.handlers.KaggleDownloader import KaggleDownloader
from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class KaggleLhqDownloader(KaggleDownloader):
    """Handler to download LHQ dataset from Kaggle and extract only images from a specific category.

    This handler extends KaggleDownloader to filter and extract only images
    from a specific category based on filename suffixes (e.g., 00000000_(2).jpg).

    File naming convention:
        - 00000000.jpg -> default category (no suffix)
        - 00000000_(2).jpg -> category 2
        - 00000000_(3).jpg -> category 3
        - etc.

    Requires:
        - kaggle package installed (pip install kaggle)

    Attributes:
        dataset_id (str): Unique identifier for the dataset
        kaggle_dataset (str): Kaggle dataset identifier (e.g., 'arnaud58/landscape-pictures')
        provided_destination_path (Path): Destination path for extracted files
        category (str): Category name to filter (e.g., 'mountain')
        category_map (dict): Mapping from category names to filename suffixes (e.g., {'mountain': '2', 'beach': '3'})
        target_hash (str, optional): Expected hash value for verification
    """

    def __init__(self, dataset_id: str, kaggle_dataset: str, provided_destination_path: Path,
                 archive_root: str, category: str = "mountain", category_map: dict = None, target_hash: str = None):
        super().__init__(dataset_id, kaggle_dataset, provided_destination_path, archive_root, target_hash)
        self.category = category.lower()
        self.category_map = category_map or {}

        # Get the suffix for this category from the map
        self.category_suffix = self.category_map.get(self.category, None)

    def _matches_category_suffix(self, filename: str) -> bool:
        """Check if filename matches the category suffix pattern.

        Args:
            filename: Name of the file to check

        Returns:
            True if the file matches the category, False otherwise
        """
        # If no suffix mapping, match files without suffix (default category)
        if self.category_suffix is None:
            # Match files like 00000000.jpg (no suffix)
            return not re.search(r'_\(\d+\)', filename)

        # Match files with specific suffix like 00000000_(2).jpg
        pattern = rf'_\({re.escape(self.category_suffix)}\)\.'
        return bool(re.search(pattern, filename))

    def _extract_category_images(self, zip_file: Path) -> None:
        """Extract only images from a specific category based on filename suffixes.

        Args:
            zip_file: Path to the downloaded zip file
        """
        os.makedirs(self.destination_path, exist_ok=True)
        abs_dest = os.path.abspath(self.destination_path)

        extracted_count = 0
        skipped_count = 0

        suffix_desc = "default (no suffix)" if self.category_suffix is None else f"suffix '({self.category_suffix})'"
        logger.info(f"Extracting {self.category} images ({suffix_desc}) from {zip_file} to {abs_dest}")

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                all_files = zip_ref.namelist()

                logger.debug(f"Total files in zip: {len(all_files)}")
                logger.debug(f"First 5 files: {all_files[:5]}")

                # Filter for category images
                category_files = []
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

                # Collect statistics about suffixes for debugging
                suffix_stats = {}

                for file in all_files:
                    if file.endswith('/'):
                        continue

                    filename = os.path.basename(file)

                    # Track suffix statistics
                    suffix_match = re.search(r'_\((\d+)\)', filename)
                    if suffix_match:
                        suffix = suffix_match.group(1)
                        suffix_stats[suffix] = suffix_stats.get(suffix, 0) + 1
                    else:
                        suffix_stats['default'] = suffix_stats.get('default', 0) + 1

                    # Check if file is an image and matches category suffix
                    if file.lower().endswith(image_extensions):
                        if self._matches_category_suffix(filename):
                            category_files.append(file)

                if not category_files:
                    logger.warning(f"No files found with category '{self.category}' ({suffix_desc})")
                    logger.info(f"Suffix distribution in dataset: {suffix_stats}")
                    logger.info(f"Sample image files: {[f for f in all_files[:10] if f.lower().endswith(image_extensions)]}")
                    raise ValueError(f"No {self.category} images found in dataset. Expected {suffix_desc}. Suffix distribution: {suffix_stats}")

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
