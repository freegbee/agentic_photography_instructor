from pathlib import Path

import cv2
from numpy import ndarray


class ImageUtils:

    @staticmethod
    def load_image(image_path: str) -> ndarray:
        """
        Load an image from the specified path and return it as a numpy ndarray.
        Raises:
            ValueError: If image_path is empty or not a string.
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the path points to a directory.
            IOError: If OpenCV fails to load the image.
        """
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("image_path must be a non-empty string")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Image path is not a file: {image_path}")

        image: ndarray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"Failed to load image (cv2 returned None): {image_path}")
        return image

    @staticmethod
    def save_image(image: ndarray, image_path: str) -> None:
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(image_path, image)
        if not success:
            raise IOError(f"Failed to write image to: {image_path}")
