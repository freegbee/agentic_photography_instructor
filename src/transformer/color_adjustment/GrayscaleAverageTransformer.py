import numpy as np
from numpy import ndarray

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class GrayscaleAverageTransformer(AbstractColorAdjustmentTransformer):
    label = "CA_GRAY_AVG"
    description = "Converts the image to grayscale by averaging the color channels."

    def transform(self, image: ndarray) -> ndarray:
        # Vectorized implementation avoids overflow and loop overhead
        # Calculate mean across channels (axis 2) using float to prevent uint8 overflow
        avg = np.mean(image, axis=2, dtype=float).astype(image.dtype)
        
        # Reconstruct 3-channel image (BGR)
        return np.dstack((avg, avg, avg))
