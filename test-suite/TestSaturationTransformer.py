import cv2
import numpy as np
import pytest

from transformer.color_adjustment.saturation_transformer import (
    SaturationIncreaseTransformerWeak,
    SaturationIncreaseTransformerMedium,
    SaturationIncreaseTransformerStrong,
    SaturationDecreaseTransformerWeak,
    SaturationDecreaseTransformerMedium,
    SaturationDecreaseTransformerStrong
)


def _get_mean_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 1])


def test_saturation_increase_increases_saturation():
    # Create an image with some color but not fully saturated
    # 50x50 pixels, pure red but with medium saturation (e.g. R=150, G=50, B=50)
    image = np.full((50, 50, 3), [50, 50, 150], dtype=np.uint8)
    sat_orig = _get_mean_saturation(image)

    transformers = [
        SaturationIncreaseTransformerWeak(),
        SaturationIncreaseTransformerMedium(),
        SaturationIncreaseTransformerStrong()
    ]

    for t in transformers:
        out = t.transform(image)
        sat_new = _get_mean_saturation(out)
        assert sat_new > sat_orig, f"{t.label} did not increase saturation"
        assert out.shape == image.shape


def test_saturation_decrease_decreases_saturation():
    # Create a highly saturated image
    image = np.full((50, 50, 3), [10, 10, 200], dtype=np.uint8)
    sat_orig = _get_mean_saturation(image)

    transformers = [
        SaturationDecreaseTransformerWeak(),
        SaturationDecreaseTransformerMedium(),
        SaturationDecreaseTransformerStrong()
    ]

    for t in transformers:
        out = t.transform(image)
        sat_new = _get_mean_saturation(out)
        assert sat_new < sat_orig, f"{t.label} did not decrease saturation"
        assert out.shape == image.shape