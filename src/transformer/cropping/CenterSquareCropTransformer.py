from numpy import ndarray

from transformer.cropping.AbstractCroppingTransformer import AbstractCroppingTransformer


class CenterSquareCropTransformer(AbstractCroppingTransformer):
    def __init__(self):
        super().__init__("C1-1_Center", "Creates a squared image centered on the image center.")

    def transform(self, image: ndarray) -> ndarray:
        height, width = image.shape[:2]
        min_dim = min(height, width)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        return image[start_y:start_y + min_dim, start_x:start_x + min_dim]