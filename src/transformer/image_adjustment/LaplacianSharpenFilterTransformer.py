import cv2
import numpy as np

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import AbstractImageAdjustmentTransformer


class LaplacianSharpenFilterTransformer(AbstractImageAdjustmentTransformer):

    label = "UA_LAPLACIAN_1"
    description = "Sharpen image with Laplacian filter"

    def transform(self, image: np.ndarray) -> np.ndarray:
        # see https://www.opencvhelp.org/tutorials/image-processing/how-to-sharpen-image/
        sigma = 1.0  # Standard deviation for Gaussian kernel
        strength = 1.5  # Strength of the sharpening effect
        kernel_size = (5, 5)  # Kernel size for Gaussian blur
        # Apply Gaussian blur with specified kernel size
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        # Subtract the blurred image from the original
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened
