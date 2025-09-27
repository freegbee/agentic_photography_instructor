import numpy as np

from transformer.cropping.AbstractCroppingTransformer import AbstractCroppingTransformer


class NoCropTransformer(AbstractCroppingTransformer):

    label = "C-NONE"
    description = "Does not crop the image."

    def transform(self, image: np.ndarray) -> np.ndarray:
        return image.copy()