from pathlib import Path

import numpy as np


class ImageUtils:
    @staticmethod
    def load_image_from_path(image_path: Path | str) -> np.ndarray:
        """
        Loads an image from the specified path and returns it as a numpy ndarray. The color order is BGR, as cv2 loads images in BGR format by default.
        :param image_path:
        :return:
        """
        import cv2
        image: np.ndarray = cv2.imread(image_path)
        return image

    @staticmethod
    def save_image_to_path(image: np.ndarray, image_path: Path | str) -> None:
        import cv2
        cv2.imwrite(image_path, image)

    @staticmethod
    def resize_to_max_dimensions(image: np.ndarray, max_size) -> np.ndarray:
        """ Resizes the image to fit within a square of max_size x max_size, preserving the aspect ratio."""
        import cv2
        height, width = image.shape[:2]
        if height > max_size or width > max_size:
            scaling_factor = max_size / float(max(height, width))
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image