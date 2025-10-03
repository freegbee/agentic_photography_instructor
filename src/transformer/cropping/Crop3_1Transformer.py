from numpy import ndarray

from transformer.cropping.AbstractCroppingTransformer import AbstractCroppingTransformer


class Crop3_1Transformer(AbstractCroppingTransformer):

    label = "C3_1_Center"
    description = "Creates an image in 3*1 format centered on the image center."

    def transform(self, image: ndarray) -> ndarray:
        target_aspect_ratio = 3 / 1

        return self._crop_to_aspect_ratio(image, target_aspect_ratio)
