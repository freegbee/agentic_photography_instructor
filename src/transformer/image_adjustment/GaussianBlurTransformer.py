import cv2
from numpy import ndarray

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import AbstractImageAdjustmentTransformer


class GaussianBlurTransformer3(AbstractImageAdjustmentTransformer):
    """Apply gaussian blur with ksize=3 to the image."""

    label = "IA_BLUR_GAU_3"
    description = "Apply gaussian blur with k=3."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.GaussianBlur(src=image, ksize=(3, 3), sigmaX=1)

class GaussianBlurTransformer5(AbstractImageAdjustmentTransformer):
    """Apply gaussian blur with ksize=5 to the image."""

    label = "IA_BLUR_GAU_5"
    description = "Apply gaussian blur with k=5."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=1)


class GaussianBlurTransformer7(AbstractImageAdjustmentTransformer):
    """Apply gaussian blur with ksize=7 to the image."""

    label = "IA_BLUR_GAU_7"
    description = "Apply gaussian blur with k=7."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.GaussianBlur(src=image, ksize=(7, 7), sigmaX=1)

class GaussianBlurTransformer9(AbstractImageAdjustmentTransformer):
    """Apply gaussian blur with ksize=9 to the image."""

    label = "IA_BLUR_GAU_9"
    description = "Apply gaussian blur with k=9."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.GaussianBlur(src=image, ksize=(9, 9), sigmaX=1)

class GaussianBlurTransformer11(AbstractImageAdjustmentTransformer):
    """Apply gaussian blur with ksize=11 to the image."""

    label = "IA_BLUR_GAU_11"
    description = "Apply gaussian blur with k=11."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.GaussianBlur(src=image, ksize=(11, 11), sigmaX=1)