from numpy import ndarray

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class GrayscaleAverageTransformer(AbstractColorAdjustmentTransformer):
    def __init__(self):
        super().__init__("grayscale_average", "Transforms Image to grayscale using average method")

    def transform(self, image: ndarray) -> ndarray:
        # see https://www.geeksforgeeks.org/python/python-grayscaling-of-images-using-opencv/, method 3
        (row, col) = image.shape[0:2]

        # Take the average of pixel values of the BGR Channels
        # to convert the colored image to grayscale image
        for i in range(row):
            for j in range(col):
                # Find the average of the BGR pixel values
                image[i, j] = sum(image[i, j]) * 0.33

        return image
