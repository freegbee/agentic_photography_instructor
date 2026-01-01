from abc import abstractmethod
import cv2
import numpy as np
from numpy import ndarray

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import AbstractImageAdjustmentTransformer


class AbstractSharpenTransformer(AbstractImageAdjustmentTransformer):
    """Base class for sharpening transformers using convolution kernels."""

    @property
    @abstractmethod
    def kernel(self) -> np.ndarray:
        pass

    def transform(self, image: ndarray) -> ndarray:
        # ddepth=-1 keeps the same depth as source (uint8), handling clipping automatically
        return cv2.filter2D(src=image, ddepth=-1, kernel=self.kernel)


class SharpenTransformerWeak(AbstractSharpenTransformer):
    """Apply weak sharpening using a 3x3 kernel."""
    label = "IA_SHARPEN_WEAK"
    description = "Apply weak sharpening."

    @property
    def kernel(self) -> np.ndarray:
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


class SharpenTransformerStrong(AbstractSharpenTransformer):
    """Apply strong sharpening using a 3x3 kernel including diagonals."""
    label = "IA_SHARPEN_STRONG"
    description = "Apply strong sharpening."

    @property
    def kernel(self) -> np.ndarray:
        return np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)