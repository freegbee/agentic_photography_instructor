import cv2
import numpy as np

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class BrightnessAndContrastTransformer(AbstractColorAdjustmentTransformer):

    label = "CA_BRIGHT_CONTRAST"
    description = "Adjusts the brighness and contrast  of an image"

    @staticmethod
    def __convert_scale(image: np.ndarray, alpha, beta) -> np.ndarray:
        """Add bias and gain to an image with saturation arithmetics. Unlike
           cv2.convertScaleAbs, it does not take an absolute value, which would lead to
           nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
           becomes 78 with OpenCV, when in fact it should become 0).
        """
        new_img = image * alpha + beta
        new_img[new_img < 0] = 0
        new_img[new_img > 255] = 255
        return new_img.astype(np.uint8)

    def transform(self, image: np.ndarray) -> np.ndarray:
        clip_hist_percent = 25
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = [float(hist[0])]
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        '''
        # Calculate new histogram with desired range and show histogram 
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        plt.show()
        '''

        auto_result = self.__convert_scale(image, alpha=alpha, beta=beta)
        return auto_result