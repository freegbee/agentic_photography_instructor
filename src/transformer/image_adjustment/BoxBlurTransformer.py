import cv2
import numpy as np
from numpy import ndarray

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import AbstractImageAdjustmentTransformer

def box_blur(image: ndarray, ksize: int) -> ndarray:
    """Apply box blur to the image using a normalized box filter."""
    box_blur_kernel = np.ones((ksize, ksize), np.float32) / ksize**2
    return cv2.filter2D(src=image, ddepth=-1, kernel=box_blur_kernel)

class BoxBlurTransformer3(AbstractImageAdjustmentTransformer):
    """Apply box blur with ksize=3 to the image."""

    label = "IA_BLUR_BOX3"
    description = "Apply box blur with k=3."
    

    def transform(self, image: ndarray) -> ndarray:
        return box_blur(image, 3)

class BoxBlurTransformer5(AbstractImageAdjustmentTransformer):
    """Apply box blur with ksize=5 to the image."""

    label = "IA_BLUR_BOX5"
    description = "Apply box blur with k=5."

    def transform(self, image: ndarray) -> ndarray:
        return box_blur(image, 5)


class BoxBlurTransformer7(AbstractImageAdjustmentTransformer):
    """Apply box blur with ksize=7 to the image."""

    label = "IA_BLUR_BOX7"
    description = "Apply box blur with k=7."

    def transform(self, image: ndarray) -> ndarray:
        return box_blur(image, 7)

class BoxBlurTransformer9(AbstractImageAdjustmentTransformer):
    """Apply box blur with ksize=9 to the image."""

    label = "IA_BLUR_BOX9"
    description = "Apply box blur with k=9."

    def transform(self, image: ndarray) -> ndarray:
        return box_blur(image, 8)

class BoxBlurTransformer11(AbstractImageAdjustmentTransformer):
    """Apply box blur with ksize=11 to the image."""

    label = "IA_BLUR_BOX11"
    description = "Apply box blur with k=11."

    def transform(self, image: ndarray) -> ndarray:
        return box_blur(image, 11)