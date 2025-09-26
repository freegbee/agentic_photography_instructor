import cv2
from numpy import ndarray

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import AbstractImageAdjustmentTransformer


class MedianBlurTransformer3(AbstractImageAdjustmentTransformer):
    """Apply median blur with ksize=3 to the image."""

    label = "IA_BLUR_MED_3"
    description = "Apply median blur with k=3."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.medianBlur(src=image, ksize=3)

class MedianBlurTransformer5(AbstractImageAdjustmentTransformer):
    """Apply median blur with ksize=5 to the image."""

    label = "IA_BLUR_MED_5"
    description = "Apply median blur with k=5."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.medianBlur(src=image, ksize=5)


class MedianBlurTransformer7(AbstractImageAdjustmentTransformer):
    """Apply median blur with ksize=7 to the image."""

    label = "IA_BLUR_MED_7"
    description = "Apply median blur with k=7."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.medianBlur(src=image, ksize=7)

class MedianBlurTransformer9(AbstractImageAdjustmentTransformer):
    """Apply median blur with ksize=9 to the image."""

    label = "IA_BLUR_MED_9"
    description = "Apply median blur with k=9."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.medianBlur(src=image, ksize=9)

class MedianBlurTransformer11(AbstractImageAdjustmentTransformer):
    """Apply median blur with ksize=7 to the image."""

    label = "IA_BLUR_MED_11"
    description = "Apply median blur with k=9."

    def transform(self, image: ndarray) -> ndarray:
        return cv2.medianBlur(src=image, ksize=11)