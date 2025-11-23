import numpy as np
import pytest

from transformer.image_adjustment.ShiftColorPlaneTransformer import (
    ShiftColorPlaneLeftB,
    ShiftColorPlaneLeftG,
    ShiftColorPlaneLeftR,
    ShiftColorPlaneDownB,
    ShiftColorPlaneDownG,
    ShiftColorPlaneDownR,
    ShiftColorPlaneLeftAll,
    ShiftColorPlaneDownAll,
)


@pytest.mark.parametrize(
    "transformer_cls, channel_idx, direction",
    [
        (ShiftColorPlaneLeftB, 0, "left"),
        (ShiftColorPlaneLeftG, 1, "left"),
        (ShiftColorPlaneLeftR, 2, "left"),
        (ShiftColorPlaneDownB, 0, "down"),
        (ShiftColorPlaneDownG, 1, "down"),
        (ShiftColorPlaneDownR, 2, "down"),
        (ShiftColorPlaneLeftAll, None, "left"),
        (ShiftColorPlaneDownAll, None, "down"),
    ],
)
def test_shift_color_plane_basic(transformer_cls, channel_idx, direction):
    # Create a small deterministic image HxW=4x6 with distinct per-channel values
    h, w = 4, 6
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        # fill channel with unique values so we can detect shifts
        img[..., c] = (c + 1) * 10 + np.arange(h).reshape(h, 1) * 10 + np.arange(w)

    original = img.copy()

    transformer = transformer_cls()
    out = transformer.transform(img)

    # original must not be mutated
    assert np.array_equal(img, original)

    # other channels unchanged (for single-channel transforms), or all channels tested for ALL
    if channel_idx is not None:
        for c in range(3):
            if c != channel_idx:
                assert np.array_equal(out[..., c], original[..., c])

    # compute expected using circular roll semantics
    expected = original.copy()
    shift_h = h // 2
    shift_w = w // 2
    if direction == "left":
        if channel_idx is None:
            expected = np.roll(original, -shift_w, axis=1)
        else:
            expected[..., channel_idx] = np.roll(original[..., channel_idx], -shift_w, axis=1)
    else:  # down
        if channel_idx is None:
            expected = np.roll(original, shift_h, axis=0)
        else:
            expected[..., channel_idx] = np.roll(original[..., channel_idx], shift_h, axis=0)

    assert np.array_equal(out, expected)


def test_shift_color_plane_invalid_input_shape():
    # 2D image
    arr2d = np.zeros((4, 4), dtype=np.uint8)
    t = ShiftColorPlaneLeftB()
    with pytest.raises(ValueError):
        t.transform(arr2d)

    # 4-channel image
    arr4 = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        t.transform(arr4)
