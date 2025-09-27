import numpy as np

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class NoColorAdjustmentTransformer(AbstractColorAdjustmentTransformer):

    label = "CA_NONE"
    description = "Transformer that does not transform the color of the image"

    def transform(self, image: np.ndarray) -> np.ndarray:
        # Keep 3 channels but make it grayscale. No transformer may change the number of channels or the color channel order.
        return image.copy()