from abc import abstractmethod
import cv2
import numpy as np
from numpy import ndarray

from transformer.lighting.abstract_lighting_transformer import AbstractLightingTransformer


class AbstractContrastTransformer(AbstractLightingTransformer):
    """Base class for contrast adjustments centered around gray value 127.5."""

    @property
    @abstractmethod
    def alpha(self) -> float:
        pass

    def transform(self, image: ndarray) -> ndarray:
        # Formula: new_val = alpha * (old_val - 127.5) + 127.5
        # This expands/contracts the histogram around the center.
        # cv2.convertScaleAbs calculates: alpha * src + beta
        # So: alpha * src - 127.5 * alpha + 127.5
        # beta = 127.5 * (1.0 - alpha)
        beta = 127.5 * (1.0 - self.alpha)
        return cv2.convertScaleAbs(image, alpha=self.alpha, beta=beta)


class ContrastIncreaseTransformerWeak(AbstractContrastTransformer):
    label = "LI_CONTRAST_INC_WEAK"
    description = "Increase contrast weakly (alpha=1.2)."
    alpha = 1.2
    reverse_transformer_label = "LI_CONTRAST_DEC_WEAK"


class ContrastIncreaseTransformerMedium(AbstractContrastTransformer):
    label = "LI_CONTRAST_INC_MED"
    description = "Increase contrast medium (alpha=1.5)."
    alpha = 1.5
    reverse_transformer_label = "LI_CONTRAST_DEC_MED"


class ContrastIncreaseTransformerStrong(AbstractContrastTransformer):
    label = "LI_CONTRAST_INC_STRONG"
    description = "Increase contrast strongly (alpha=2.0)."
    alpha = 2.0
    reverse_transformer_label = "LI_CONTRAST_DEC_STRONG"


class ContrastDecreaseTransformerWeak(AbstractContrastTransformer):
    label = "LI_CONTRAST_DEC_WEAK"
    description = "Decrease contrast weakly (alpha=0.8)."
    alpha = 0.8
    reverse_transformer_label = "LI_CONTRAST_INC_WEAK"


class ContrastDecreaseTransformerMedium(AbstractContrastTransformer):
    label = "LI_CONTRAST_DEC_MED"
    description = "Decrease contrast medium (alpha=0.6)."
    alpha = 0.6
    reverse_transformer_label = "LI_CONTRAST_INC_MED"


class ContrastDecreaseTransformerStrong(AbstractContrastTransformer):
    label = "LI_CONTRAST_DEC_STRONG"
    description = "Decrease contrast strongly (alpha=0.5)."
    alpha = 0.5
    reverse_transformer_label = "LI_CONTRAST_INC_STRONG"


class CLAHETransformer(AbstractLightingTransformer):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    label = "LI_CLAHE"
    description = "Adaptive histogram equalization on L-channel."

    def transform(self, image: ndarray) -> ndarray:
        # Convert to LAB color space to operate only on Lightness channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge and convert back to BGR
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

class AutoContrastTransformer(AbstractLightingTransformer):
    """Optimize dynamic range by stretching histogram to full range."""

    label = "LI_AUTO_CONTRAST"
    description = "Automatically optimize dynamic range with histogram stretching"

    def transform(self, image: ndarray) -> ndarray:
        """Apply auto contrast by stretching each channel independently."""
        result = np.zeros_like(image)

        # Process each channel independently
        for i in range(3):  # B, G, R channels
            channel = image[:, :, i]

            # Find min and max values in a channel
            min_val = np.min(channel)
            max_val = np.max(channel)

            # Avoid division by zero
            if max_val - min_val > 0:
                # Stretch to full 0-255 range
                result[:, :, i] = np.clip(
                    255.0 * (channel.astype(np.float32) - min_val) / (max_val - min_val),
                    0, 255
                ).astype(np.uint8)
            else:
                result[:, :, i] = channel

        return result