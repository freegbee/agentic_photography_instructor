import cv2
from numpy import ndarray

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class GrayscaleTransformer(AbstractColorAdjustmentTransformer):

    label = "CA_GRAY"
    description = "Transforms Image to grayscales."

    def transform(self, image: ndarray) -> ndarray:
        # Keep 3 channels but make it grayscale. No transformer may change the number of channels or the color channel order.
        return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
