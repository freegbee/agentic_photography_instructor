from numpy import ndarray
from pathlib import Path


class TestingUtils:
    @staticmethod
    def load_image_from_path(image_path: Path | str) -> ndarray:
        import cv2
        image: ndarray = cv2.imread(image_path)
        return image

    @staticmethod
    def save_image_to_path(image: ndarray, image_path: Path | str) -> None:
        import cv2
        cv2.imwrite(image_path, image)

