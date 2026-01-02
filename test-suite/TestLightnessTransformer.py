import numpy as np
import pytest

from transformer.lighting.lightness_transformer import (
    LightnessIncreaseTransformerWeak,
    LightnessIncreaseTransformerMedium,
    LightnessIncreaseTransformerStrong,
    LightnessDecreaseTransformerWeak,
    LightnessDecreaseTransformerMedium,
    LightnessDecreaseTransformerStrong
)


def test_lightness_increase_increases_mean():
    # Create an image with mid-range values to allow increase without immediate clipping
    image = np.full((50, 50, 3), 100, dtype=np.uint8)
    mean_orig = np.mean(image)

    transformers = [
        LightnessIncreaseTransformerWeak(),
        LightnessIncreaseTransformerMedium(),
        LightnessIncreaseTransformerStrong()
    ]

    for t in transformers:
        out = t.transform(image)
        mean_new = np.mean(out)
        # Brightness increase should shift values up, increasing the mean
        assert mean_new > mean_orig, f"{t.label} did not increase brightness mean"
        assert out.shape == image.shape


def test_lightness_decrease_decreases_mean():
    # Create an image with mid-range values to allow decrease without immediate clipping
    image = np.full((50, 50, 3), 100, dtype=np.uint8)
    mean_orig = np.mean(image)

    transformers = [
        LightnessDecreaseTransformerWeak(),
        LightnessDecreaseTransformerMedium(),
        LightnessDecreaseTransformerStrong()
    ]

    for t in transformers:
        out = t.transform(image)
        mean_new = np.mean(out)
        # Brightness decrease should shift values down, decreasing the mean
        assert mean_new < mean_orig, f"{t.label} did not decrease brightness mean"
        assert out.shape == image.shape