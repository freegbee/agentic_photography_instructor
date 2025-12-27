from typing import Literal

import numpy as np
from numpy import ndarray

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import (
    AbstractImageAdjustmentTransformer,
)

Channel = Literal["B", "G", "R"]


def _shift_channel(image: ndarray, channel: Channel, direction: Literal["left", "right", "down", "up"]) -> ndarray:
    """Return a copy of the image with a single color channel circularly shifted by 50%.

    - Uses BGR channel ordering (OpenCV convention).
    - For "left" the channel content is moved left by width//2 with wrap-around.
    - For "down" the channel content is moved down by height//2 with wrap-around.
    - Preserves dtype and only modifies the specified channel; other channels are unchanged.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an HxWx3 BGR ndarray")

    h, w, _ = image.shape
    idx_map = {"B": 0, "G": 1, "R": 2}
    ch_idx = idx_map[channel]

    out = image.copy()

    if direction == "left":
        shift = w // 2
        # circular shift to the left -> negative roll along axis=1
        out[..., ch_idx] = np.roll(image[..., ch_idx], -shift, axis=1)
    elif direction == "right":
        shift = w // 2
        # circular shift to the right -> positive roll along axis=1
        out[..., ch_idx] = np.roll(image[..., ch_idx], shift, axis=1)
    elif direction == "down":
        shift = h // 2
        # circular shift down -> positive roll along axis=0
        out[..., ch_idx] = np.roll(image[..., ch_idx], shift, axis=0)
    elif direction == "up":
        shift = h // 2
        # circular shift up -> negative roll along axis=0
        out[..., ch_idx] = np.roll(image[..., ch_idx], -shift, axis=0)
    else:
        raise ValueError("direction must be 'left' or 'down'")

    return out


def _shift_all(image: ndarray, direction: Literal["left", "right", "down", "up"]) -> ndarray:
    """Return a copy of the image with all three color channels circularly shifted by 50%.

    - For "left": move content left by width//2 with wrap-around for all channels.
    - For "down": move content down by height//2 with wrap-around for all channels.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an HxWx3 BGR ndarray")

    h, w, _ = image.shape
    out = image.copy()

    if direction == "left":
        shift = w // 2
        out = np.roll(image, -shift, axis=1)
    elif direction == "right":
        shift = w // 2
        out = np.roll(image, shift, axis=1)
    elif direction == "down":
        shift = h // 2
        out = np.roll(image, shift, axis=0)
    elif direction == "up":
        shift = h // 2
        out = np.roll(image, -shift, axis=0)
    else:
        raise ValueError("direction must be 'left' or 'down'")

    return out


class ShiftColorPlaneLeftB(AbstractImageAdjustmentTransformer):
    """Shift the Blue channel left by 50% (BGR -> B shifted left)."""

    label = "IA_SHIFT_LEFT_B"
    description = "Shift the blue color plane left by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = ["IA_SHIFT_RIGHT_B"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "B", "left")


class ShiftColorPlaneRightB(AbstractImageAdjustmentTransformer):
    """Shift the Blue channel left by 50% (BGR -> B shifted left)."""

    label = "IA_SHIFT_RIGHT_B"
    description = "Shift the blue color plane right by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = ["IA_SHIFT_LEFT_B"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "B", "right")


class ShiftColorPlaneLeftG(AbstractImageAdjustmentTransformer):
    """Shift the Green channel left by 50% (BGR -> G shifted left)."""

    label = "IA_SHIFT_LEFT_G"
    description = "Shift the green color plane left by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = ["IA_SHIFT_RIGHT_G"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "G", "left")


class ShiftColorPlaneRightG(AbstractImageAdjustmentTransformer):
    """Shift the Green channel right by 50% (BGR -> G shifted right)."""

    label = "IA_SHIFT_RIGHT_G"
    description = "Shift the green color plane right by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = ["IA_SHIFT_LEFT_G"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "G", "right")


class ShiftColorPlaneLeftR(AbstractImageAdjustmentTransformer):
    """Shift the Red channel left by 50% (BGR -> R shifted left)."""

    label = "IA_SHIFT_LEFT_R"
    description = "Shift the red color plane left by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = ["IA_SHIFT_RIGHT_R"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "R", "left")


class ShiftColorPlaneRightR(AbstractImageAdjustmentTransformer):
    """Shift the Red channel left by 50% (BGR -> R shifted left)."""

    label = "IA_SHIFT_RIGHT_R"
    description = "Shift the red color plane right by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = ["IA_SHIFT_LEFT_R"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "R", "right")


class ShiftColorPlaneDownB(AbstractImageAdjustmentTransformer):
    """Shift the Blue channel down by 50% (BGR -> B shifted down)."""

    label = "IA_SHIFT_DOWN_B"
    description = "Shift the blue color plane down by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = "IA_SHIFT_UP_B"

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "B", "down")


class ShiftColorPlaneUpB(AbstractImageAdjustmentTransformer):
    """Shift the Blue channel up by 50% (BGR -> B shifted up)."""

    label = "IA_SHIFT_UP_B"
    description = "Shift the blue color plane up by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label: "IA_SHIFT_DOWN_B"

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "B", "up")


class ShiftColorPlaneDownG(AbstractImageAdjustmentTransformer):
    """Shift the Green channel down by 50% (BGR -> G shifted down)."""

    label = "IA_SHIFT_DOWN_G"
    description = "Shift the green color plane down by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = "IA_SHIFT_UP_G"

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "G", "down")


class ShiftColorPlaneUpG(AbstractImageAdjustmentTransformer):
    """Shift the Green channel up by 50% (BGR -> G shifted up)."""

    label = "IA_SHIFT_UP_G"
    description = "Shift the green color plane up by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = "IA_SHIFT_DOWN_G"

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "G", "up")


class ShiftColorPlaneDownR(AbstractImageAdjustmentTransformer):
    """Shift the Red channel down by 50% (BGR -> R shifted down)."""

    label = "IA_SHIFT_DOWN_R"
    description = "Shift the red color plane down by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = ["IA_SHIFT_UP_R"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "R", "down")


class ShiftColorPlaneUpR(AbstractImageAdjustmentTransformer):
    """Shift the Red channel up by 50% (BGR -> R shifted down)."""

    label = "IA_SHIFT_UP_R"
    description = "Shift the red color plane up by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_label = ["IA_SHIFT_DOWN_R"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "R", "up")


class ShiftColorPlaneLeftAll(AbstractImageAdjustmentTransformer):
    """Shift all three color channels left by 50% (BGR -> all channels shifted left)."""

    label = "IA_SHIFT_LEFT_ALL"
    description = "Shift all color planes left by 50% while keeping BGR format."
    reverse_transformer_label = "IA_SHIFT_RIGHT_ALL"

    def transform(self, image: ndarray) -> ndarray:
        return _shift_all(image, "left")


class ShiftColorPlaneRightAll(AbstractImageAdjustmentTransformer):
    """Shift all three color channels right by 50% (BGR -> all channels shifted right)."""

    label = "IA_SHIFT_RIGHT_ALL"
    description = "Shift all color planes right by 50% while keeping BGR format."
    reverse_transformer_labels = "IA_SHIFT_LEFT_ALL"

    def transform(self, image: ndarray) -> ndarray:
        return _shift_all(image, "right")


class ShiftColorPlaneDownAll(AbstractImageAdjustmentTransformer):
    """Shift all three color channels down by 50% (BGR -> all channels shifted down)."""

    label = "IA_SHIFT_DOWN_ALL"
    description = "Shift all color planes down by 50% while keeping BGR format."
    reverse_transformer_label = "IA_SHIFT_UP_ALL"

    def transform(self, image: ndarray) -> ndarray:
        return _shift_all(image, "down")


class ShiftColorPlaneUpAll(AbstractImageAdjustmentTransformer):
    """Shift all three color channels up by 50% (BGR -> all channels shifted up)."""

    label = "IA_SHIFT_UP_ALL"
    description = "Shift all color planes up by 50% while keeping BGR format."
    reverse_transformer_label = "IA_SHIFT_DOWN_ALL"

    def transform(self, image: ndarray) -> ndarray:
        return _shift_all(image, "up")
