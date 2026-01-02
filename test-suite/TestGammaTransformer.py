import numpy as np
import pytest

from transformer.lighting.gamma_transformer import AutoGammaCorrectionTransformer


def test_auto_gamma_brightens_dark_image():
    # Create a dark image (mean approx 50)
    image = np.full((50, 50, 3), 50, dtype=np.uint8)
    transformer = AutoGammaCorrectionTransformer()

    result = transformer.transform(image)

    # Expect result to be brighter (closer to 127.5)
    assert np.mean(result) > np.mean(image)
    # Ideally close to 127.5 (mid-gray)
    assert abs(np.mean(result) - 127.5) < 5


def test_auto_gamma_darkens_bright_image():
    # Create a bright image (mean approx 200)
    image = np.full((50, 50, 3), 200, dtype=np.uint8)
    transformer = AutoGammaCorrectionTransformer()

    result = transformer.transform(image)

    # Expect result to be darker (closer to 127.5)
    assert np.mean(result) < np.mean(image)
    assert abs(np.mean(result) - 127.5) < 5


def test_auto_gamma_handles_mid_gray_neutral():
    # Image already at mid-gray
    image = np.full((50, 50, 3), 127, dtype=np.uint8)
    transformer = AutoGammaCorrectionTransformer()

    result = transformer.transform(image)

    # Should change very little
    assert abs(np.mean(result) - np.mean(image)) < 2