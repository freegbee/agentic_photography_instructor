import math

import cv2
import numpy as np
from numpy import ndarray

from transformer.lighting.abstract_lighting_transformer import AbstractLightingTransformer


class AutoGammaCorrectionTransformer(AbstractLightingTransformer):
    """
    Applies automatic gamma correction to adjust the image brightness so that the mean intensity becomes 0.5 (mid-gray).
    """
    label = "LI_GAMMA_AUTO"
    description = "Auto gamma correction to center histogram."

    def transform(self, image: ndarray) -> ndarray:
        # Calculate mean brightness of the image (normalized to 0..1)
        mean = np.mean(image) / 255

        # Avoid extreme gamma values for very dark or very bright images (or division by zero)
        # If mean is too close to 0 or 1, gamma correction might be unstable or useless.
        if mean < 0.01 or mean > 0.99:
            return image

        # Calculate gamma such that: mean ^ gamma = 0.5
        # Therefore: gamma = log(0.5) / log(mean)
        gamma = math.log(0.5) / math.log(mean)

        # Create Lookup Table (LUT) for speed
        # Formula: output = ((input / 255) ^ gamma) * 255
        lut = (((np.arange(256) / 255.0) ** gamma) * 255).astype(np.uint8)

        # Apply LUT
        return cv2.LUT(image, lut)

class GammaBrightenTransformer(AbstractLightingTransformer):
    """Brighten image slightly using gamma correction (gamma=0.8)."""

    label = "LI_GAMMA_0.8"
    description = "Brighten image slightly with gamma correction (0.8)"

    def transform(self, image: ndarray) -> ndarray:
        """Apply gamma correction with gamma=0.8 to brighten."""
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / 0.8
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

        # Apply gamma correction using lookup table
        return cv2.LUT(image, table)


class GammaDarkenTransformer(AbstractLightingTransformer):
    """Darken image slightly using gamma correction (gamma=1.2)."""

    label = "LI_GAMMA_1.2"
    description = "Darken image slightly with gamma correction (1.2)"

    def transform(self, image: ndarray) -> ndarray:
        """Apply gamma correction with gamma=1.2 to darken."""
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / 1.2
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

        # Apply gamma correction using lookup table
        return cv2.LUT(image, table)
