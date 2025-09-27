import numpy as np

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import AbstractImageAdjustmentTransformer


class NoImageAdjustmentTransformer(AbstractImageAdjustmentTransformer):
    """Apply median blur with ksize=3 to the image."""

    label = "IA_NONE"
    description = "Apply no image adjustment."

    def transform(self, image: np.ndarray) -> np.ndarray:
        return image.copy()
