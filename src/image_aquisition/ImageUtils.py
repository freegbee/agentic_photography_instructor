from pathlib import Path

from numpy import ndarray


class ImageUtils:
    @staticmethod
    def load_image_from_path(image_path: Path | str) -> ndarray:
        """
        Loads an image from the specified path and returns it as a numpy ndarray. The color order is BGR, as cv2 loads images in BGR format by default.
        :param image_path:
        :return:
        """
        import cv2
        image: ndarray = cv2.imread(image_path)
        return image

    @staticmethod
    def save_image_to_path(image: ndarray, image_path: Path | str) -> None:
        import cv2
        cv2.imwrite(image_path, image)
