import cv2
import numpy as np
import pytest

from transformer.color_adjustment.saturation_transformer import (
    SaturationIncreaseTransformerWeak,
    SaturationIncreaseTransformerMedium,
    SaturationIncreaseTransformerStrong,
    SaturationDecreaseTransformerWeak,
    SaturationDecreaseTransformerMedium,
    SaturationDecreaseTransformerStrong,
    VibranceTransformer
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


def test_vibrance_boosts_low_saturation_more_than_high():
    # Create two images: one with low saturation, one with high
    # Low sat: (100, 100, 120) -> diff 20
    low_sat_img = np.full((10, 10, 3), [100, 100, 120], dtype=np.uint8)
    # High sat: (50, 50, 200) -> diff 150
    high_sat_img = np.full((10, 10, 3), [50, 50, 200], dtype=np.uint8)

    transformer = VibranceTransformer()

    low_out = transformer.transform(low_sat_img)
    high_out = transformer.transform(high_sat_img)

    low_gain = _get_mean_saturation(low_out) - _get_mean_saturation(low_sat_img)
    high_gain = _get_mean_saturation(high_out) - _get_mean_saturation(high_sat_img)

    # Vibrance should boost the dull image significantly more than the already vivid one
    assert low_gain > 0
    assert high_gain >= 0
    assert low_gain > high_gain