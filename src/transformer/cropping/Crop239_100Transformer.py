from numpy import ndarray

from transformer.cropping.AbstractCroppingTransformer import AbstractCroppingTransformer


class Crop239_100Transformer(AbstractCroppingTransformer):

    label = "C239_100_Center"
    description = "Creates an image in 239*100 format centered on the image center."

    def transform(self, image: ndarray) -> ndarray:
        target_aspect_ratio = 2 / 1

        return self._crop_to_aspect_ratio(image, target_aspect_ratio)
