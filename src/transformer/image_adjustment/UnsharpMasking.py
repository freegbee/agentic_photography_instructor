import cv2
import numpy as np

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import AbstractImageAdjustmentTransformer


class UnsharpMaskingTransformer(AbstractImageAdjustmentTransformer):

    label = "UA_UNSHARP_MASK_1"
    description = "Unsharp Masking Transform"

    def transform(self, image: np.ndarray) -> np.ndarray:
        # see https://www.opencvhelp.org/tutorials/image-processing/how-to-sharpen-image/
        sigma = 1.0  # Standard deviation for Gaussian kernel
        strength = 1.5  # Strength of the sharpening effect
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        # Subtract the blurred image from the original
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened
