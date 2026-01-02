import numpy as np
import pytest

from transformer.lighting.contrast_transformer import (
    ContrastIncreaseTransformerWeak,
    ContrastIncreaseTransformerMedium,
    ContrastIncreaseTransformerStrong,
    ContrastDecreaseTransformerWeak,
    ContrastDecreaseTransformerMedium,
    ContrastDecreaseTransformerStrong
)


def test_contrast_increase_increases_std_dev():
    # Create an image with moderate variance centered around 128
    # 50x50 pixels, random values between 100 and 155
    image = np.random.randint(100, 155, (50, 50, 3), dtype=np.uint8)
    std_orig = np.std(image)

    transformers = [
        ContrastIncreaseTransformerWeak(),
        ContrastIncreaseTransformerMedium(),
        ContrastIncreaseTransformerStrong()
    ]

    for t in transformers:
        out = t.transform(image)
        std_new = np.std(out)
        # Contrast increase should spread the histogram, increasing standard deviation
        assert std_new > std_orig, f"{t.label} did not increase contrast standard deviation"
        assert out.shape == image.shape


def test_contrast_decrease_decreases_std_dev():
    # Create an image with high variance (full range)
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    std_orig = np.std(image)

    transformers = [
        ContrastDecreaseTransformerWeak(),
        ContrastDecreaseTransformerMedium(),
        ContrastDecreaseTransformerStrong()
    ]

    for t in transformers:
        out = t.transform(image)
        std_new = np.std(out)
        # Contrast decrease should compress the histogram, decreasing standard deviation
        assert std_new < std_orig, f"{t.label} did not decrease contrast standard deviation"
        assert out.shape == image.shape