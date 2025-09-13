import cv2
from numpy import ndarray

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class GrayscaleTransformer(AbstractColorAdjustmentTransformer):

    label = "CA_GRAY"
    description = "Transforms Image to grayscales."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
