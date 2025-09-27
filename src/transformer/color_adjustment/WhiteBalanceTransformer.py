import cv2
import numpy as np

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class WhiteBalanceTransformer(AbstractColorAdjustmentTransformer):

    label = "CA_WHITEBALANCE"
    description = "Adjusts the white balance of an image"

    def transform(self, image: np.ndarray) -> np.ndarray:
        # see https://stackoverflow.com/a/46391574
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result