import cv2
import numpy as np
import pytest

from transformer.image_adjustment.SharpenTransformer import SharpenTransformerWeak, SharpenTransformerStrong


def test_sharpen_weak_changes_image():
    # Create an image with edges (rectangle)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.rectangle(image, (5, 5), (15, 15), (100, 100, 100), -1)

    transformer = SharpenTransformerWeak()
    out = transformer.transform(image)

    # Sharpening should change pixel values around edges
    assert not np.array_equal(out, image)
    assert out.shape == image.shape


def test_sharpen_strong_has_stronger_effect():
    # Create an image with edges
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.rectangle(image, (5, 5), (15, 15), (100, 100, 100), -1)

    t_weak = SharpenTransformerWeak()
    t_strong = SharpenTransformerStrong()

    out_weak = t_weak.transform(image)
    out_strong = t_strong.transform(image)

    # Calculate absolute difference sum from original
    diff_weak = np.sum(np.abs(out_weak.astype(int) - image.astype(int)))
    diff_strong = np.sum(np.abs(out_strong.astype(int) - image.astype(int)))

    # Strong sharpening should result in greater deviation from the original image
    assert diff_strong > diff_weak
    assert out_strong.shape == image.shape