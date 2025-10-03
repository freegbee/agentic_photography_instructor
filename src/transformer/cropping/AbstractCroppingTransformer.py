from abc import abstractmethod

from numpy import ndarray

from transformer.AbstractTransformer import AbstractTransformer
from transformer.TransformerTypes import TransformerTypeEnum


class AbstractCroppingTransformer(AbstractTransformer):
    """
    Abstract class for cropping transformers.

    See https://opencv.org/blog/cropping-an-image-using-opencv/ for inspiration.

    See https://www.proglobalbusinesssolutions.com/photo-aspect-ratio/ on aspect ratios.
    """

    transformer_type = TransformerTypeEnum.CROP

    @abstractmethod
    def transform(self, data: ndarray) -> ndarray:
        pass

    def _crop_to_aspect_ratio(self, image: ndarray, target_aspect_ratio: float) -> ndarray:
        height, width = image.shape[:2]

        if width / height > target_aspect_ratio:
            # Image is too wide, crop the sides
            new_width = int(height * target_aspect_ratio)
            start_x = (width - new_width) // 2
            return image[:, start_x:start_x + new_width]
        else:
            # Image is too tall, crop the top and bottom
            new_height = int(width / target_aspect_ratio)
            start_y = (height - new_height) // 2
            return image[start_y:start_y + new_height, :]