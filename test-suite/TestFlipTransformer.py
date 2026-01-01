import numpy as np
import pytest

from transformer.image_adjustment.FlipTransformer import FlipHorizontalTransformer, FlipVerticalTransformer


def test_flip_horizontal_logic_and_reversibility():
    # Create a 2x3 image (Height x Width x Channels)
    # Row 0: [10,10,10], [20,20,20], [30,30,30]
    # Row 1: [40,40,40], [50,50,50], [60,60,60]
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = 10
    image[0, 1] = 20
    image[0, 2] = 30
    image[1, 0] = 40
    image[1, 1] = 50
    image[1, 2] = 60

    transformer = FlipHorizontalTransformer()

    # 1. Transform
    flipped = transformer.transform(image)

    # Check logic: Horizontal flip mirrors columns.
    # Row 0 should be: [30], [20], [10]
    assert np.array_equal(flipped[0, 0], [30, 30, 30])
    assert np.array_equal(flipped[0, 2], [10, 10, 10])

    # 2. Reverse (Apply again)
    restored = transformer.transform(flipped)

    # Check reversibility
    assert np.array_equal(restored, image)


def test_flip_vertical_logic_and_reversibility():
    # Create a 3x2 image
    # Col 0: 10, 20, 30
    # Col 1: 40, 50, 60
    image = np.zeros((3, 2, 3), dtype=np.uint8)
    image[0, 0] = 10
    image[1, 0] = 20
    image[2, 0] = 30
    image[0, 1] = 40
    image[1, 1] = 50
    image[2, 1] = 60

    transformer = FlipVerticalTransformer()

    # 1. Transform
    flipped = transformer.transform(image)

    # Check logic: Vertical flip mirrors rows.
    # Col 0 should be: 30, 20, 10
    assert np.array_equal(flipped[0, 0], [30, 30, 30])
    assert np.array_equal(flipped[2, 0], [10, 10, 10])

    # 2. Reverse (Apply again)
    restored = transformer.transform(flipped)

    # Check reversibility
    assert np.array_equal(restored, image)