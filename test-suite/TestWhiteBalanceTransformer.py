import numpy as np
import pytest

from transformer.color_adjustment.white_balance_transformer import AutoWhiteBalanceTransformer


def test_auto_white_balance_corrects_color_cast():
    # Create an image with a strong blue cast
    # R=100, G=100, B=200
    image = np.full((50, 50, 3), [200, 100, 100], dtype=np.uint8)  # BGR

    transformer = AutoWhiteBalanceTransformer()
    result = transformer.transform(image)

    # Calculate means of channels
    b_mean = np.mean(result[:, :, 0])
    g_mean = np.mean(result[:, :, 1])
    r_mean = np.mean(result[:, :, 2])

    # In a perfectly balanced gray world, means should be equal
    # Check if they are close to each other (within margin of error due to integer arithmetic)
    assert abs(b_mean - g_mean) < 2
    assert abs(b_mean - r_mean) < 2
    assert abs(g_mean - r_mean) < 2


def test_auto_white_balance_leaves_neutral_image_mostly_intact():
    # Neutral gray image
    image = np.full((50, 50, 3), 128, dtype=np.uint8)
    transformer = AutoWhiteBalanceTransformer()
    result = transformer.transform(image)

    # Should remain roughly the same
    assert np.allclose(image, result, atol=1)