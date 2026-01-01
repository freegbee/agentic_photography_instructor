import cv2
from numpy import ndarray

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import AbstractImageAdjustmentTransformer


class FlipHorizontalTransformer(AbstractImageAdjustmentTransformer):
    """Flip the image horizontally."""

    label = "IA_FLIP_HOR"
    description = "Flip the image horizontally."
    reverse_transformer_label = "IA_FLIP_HOR"

    def transform(self, image: ndarray) -> ndarray:
        return cv2.flip(image, 1)


class FlipVerticalTransformer(AbstractImageAdjustmentTransformer):
    """Flip the image vertically."""

    label = "IA_FLIP_VERT"
    description = "Flip the image vertically."
    reverse_transformer_label = "IA_FLIP_VERT"

    def transform(self, image: ndarray) -> ndarray:
        return cv2.flip(image, 0)