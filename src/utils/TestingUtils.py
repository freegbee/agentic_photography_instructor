from numpy import ndarray
from pathlib import Path

from image_aquisition.ImageUtils import ImageUtils


class TestingUtils:
    @staticmethod
    def load_image_from_path(image_path: Path | str) -> ndarray:
        return ImageUtils.load_image_from_path(image_path)


    @staticmethod
    def save_image_to_path(image: ndarray, image_path: Path | str) -> None:
        return ImageUtils.save_image_to_path(image, image_path)

