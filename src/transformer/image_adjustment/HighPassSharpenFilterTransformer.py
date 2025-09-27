import cv2
import numpy as np

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import AbstractImageAdjustmentTransformer


class HighPassSharpenFilterTransformer(AbstractImageAdjustmentTransformer):

    label = "UA_HIGHPASS_1"
    description = "Sharpen image with Highpass filter"

    def transform(self, image: np.ndarray) -> np.ndarray:
        # see https://www.opencvhelp.org/tutorials/image-processing/how-to-sharpen-image/
        sigma = 1.0  # Standard deviation for Gaussian kernel
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        # Subtract the blurred image from the original
        high_pass = cv2.subtract(image, blurred)
        # Add the high-pass image back to the original
        sharpened = cv2.addWeighted(image, 1.0, high_pass, 1.0, 0)
        return sharpened
