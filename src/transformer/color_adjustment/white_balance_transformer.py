import cv2
import numpy as np
from numpy import ndarray

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class AutoWhiteBalanceTransformer(AbstractColorAdjustmentTransformer):
    """Applies automatic white balance using the Gray World assumption."""
    label = "CA_AUTO_WB"
    description = "Automatic White Balance (Gray World)."

    def transform(self, image: ndarray) -> ndarray:
        # Split channels
        b, g, r = cv2.split(image)

        # Calculate channel means
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)

        # Avoid division by zero (handle completely black channels)
        if b_mean == 0 or g_mean == 0 or r_mean == 0:
            return image

        # Calculate global mean (target gray value)
        k = (b_mean + g_mean + r_mean) / 3

        # Scale channels so that their mean equals the global mean
        # cv2.convertScaleAbs handles clipping and casting to uint8
        b = cv2.convertScaleAbs(b, alpha=(k / b_mean))
        g = cv2.convertScaleAbs(g, alpha=(k / g_mean))
        r = cv2.convertScaleAbs(r, alpha=(k / r_mean))

        return cv2.merge([b, g, r])