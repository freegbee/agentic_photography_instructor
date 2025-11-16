from pathlib import Path

from numpy import ndarray
import cv2


class ImageUtils:

    @staticmethod
    def load_image(image_path: str) -> ndarray:
        """
        Loads an image from the specified path and returns it as a numpy ndarray. The color order is BGR, as cv2 loads images in BGR format by default.
        :param image_path:
        :return:
        """
        image: ndarray = cv2.imread(image_path)
        return image

    @staticmethod
    def save_image(image: ndarray, image_path: str) -> None:
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(image_path, image)