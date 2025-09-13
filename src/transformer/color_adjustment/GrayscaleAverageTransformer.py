from numpy import ndarray

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class GrayscaleAverageTransformer(AbstractColorAdjustmentTransformer):
    label = "CA-GRAY-AVG"
    description = "Converts the image to grayscale by averaging the color channels."

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
