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

class UnsharpMaskLightTransformer(AbstractImageAdjustmentTransformer):
    """Apply light unsharp mask for subtle sharpening."""

    label = "IA_UNSHARP_LIGHT"
    description = "Subtle sharpening with light unsharp mask"

    def transform(self, image: ndarray) -> ndarray:
        """Apply light unsharp mask sharpening."""
        # Create Gaussian blur with larger radius for subtle effect
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

        # Light unsharp mask: original + 0.3 * (original - blurred)
        sharpened = cv2.addWeighted(image, 1.3, blurred, -0.3, 0)

        return sharpened


class ClarityTransformer(AbstractImageAdjustmentTransformer):
    """Enhance local contrast (clarity +10)."""

    label = "IA_CLARITY"
    description = "Enhance local contrast for more defined details"

    def transform(self, image: ndarray) -> ndarray:
        """Apply clarity enhancement via local contrast boost."""
        # Convert to float for processing
        img_float = image.astype(np.float32)

        # Create a blurred version (this represents local average)
        blurred = cv2.GaussianBlur(img_float, (15, 15), 3.0)

        # Calculate local contrast: difference from local average
        local_contrast = img_float - blurred

        # Boost local contrast by 10%
        clarity_boost = 0.1
        enhanced = img_float + clarity_boost * local_contrast

        # Clip and convert back to uint8
        return np.clip(enhanced, 0, 255).astype(np.uint8)
