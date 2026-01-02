"""
Color temperature transformers (warmth/coolness).

Adjusts the color temperature by shifting the blue-yellow balance.
Warm tones increase red/yellow, cool tones increase blue.
"""
import cv2
import numpy as np
from numpy import ndarray

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class WarmthTransformer(AbstractColorAdjustmentTransformer):
    """Add warm tone to image by increasing red/yellow (+15 warmth)."""

    label = "CA_WARMTH"
    description = "Add warm tone by increasing red/yellow channels"
    reverse_transformer_label = "CA_COOLNESS"

    def transform(self, image: ndarray) -> ndarray:
        """Apply warmth by boosting red and reducing blue."""
        result = image.astype(np.float32)

        # Increase red channel (index 2 in BGR)
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.15, 0, 255)

        # Decrease blue channel (index 0 in BGR)
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.85, 0, 255)

        return result.astype(np.uint8)


class CoolnessTransformer(AbstractColorAdjustmentTransformer):
    """Add cool tone to image by increasing blue (-15 warmth / +15 coolness)."""

    label = "CA_COOLNESS"
    description = "Add cool tone by increasing blue channel"
    reverse_transformer_label = "CA_WARMTH"

    def transform(self, image: ndarray) -> ndarray:
        """Apply coolness by boosting blue and reducing red."""
        result = image.astype(np.float32)

        # Increase blue channel (index 0 in BGR)
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.15, 0, 255)

        # Decrease red channel (index 2 in BGR)
        result[:, :, 2] = np.clip(result[:, :, 2] * 0.85, 0, 255)

        return result.astype(np.uint8)
