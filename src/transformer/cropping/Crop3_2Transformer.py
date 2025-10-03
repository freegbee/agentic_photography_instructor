from numpy import ndarray

from transformer.cropping.AbstractCroppingTransformer import AbstractCroppingTransformer


class Crop3_2Transformer(AbstractCroppingTransformer):

    label = "C3_2_Center"
    description = "Creates an image in 3*2 format centered on the image center."

    def transform(self, image: ndarray) -> ndarray:
        target_aspect_ratio = 3 / 2

        return self._crop_to_aspect_ratio(image, target_aspect_ratio)
