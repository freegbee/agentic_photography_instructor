from numpy import ndarray

from transformer.cropping.AbstractCroppingTransformer import AbstractCroppingTransformer


class Crop16_9Transformer(AbstractCroppingTransformer):

    label = "C16_9_Center"
    description = "Creates an image in 16*9 format centered on the image center."

    def transform(self, image: ndarray) -> ndarray:
        target_aspect_ratio = 16 / 9

        return self._crop_to_aspect_ratio(image, target_aspect_ratio)
