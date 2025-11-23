from typing import Literal, List

from numpy import ndarray
import numpy as np

from transformer.image_adjustment.AbstractImageAdjustmentTransformer import (
    AbstractImageAdjustmentTransformer,
)

Channel = Literal["B", "G", "R"]


def _shift_channel(image: ndarray, channel: Channel, direction: Literal["left", "down"]) -> ndarray:
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
    elif direction == "down":
        shift = h // 2
        # circular shift down -> positive roll along axis=0
        out[..., ch_idx] = np.roll(image[..., ch_idx], shift, axis=0)
    else:
        raise ValueError("direction must be 'left' or 'down'")

    return out


def _shift_all(image: ndarray, direction: Literal["left", "down"]) -> ndarray:
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
    elif direction == "down":
        shift = h // 2
        out = np.roll(image, shift, axis=0)
    else:
        raise ValueError("direction must be 'left' or 'down'")

    return out


class ShiftColorPlaneLeftB(AbstractImageAdjustmentTransformer):
    """Shift the Blue channel left by 50% (BGR -> B shifted left)."""

    label = "IA_SHIFT_LEFT_B"
    description = "Shift the blue color plane left by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_labels: List[str] = ["IA_SHIFT_LEFT_B"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "B", "left")


class ShiftColorPlaneLeftG(AbstractImageAdjustmentTransformer):
    """Shift the Green channel left by 50% (BGR -> G shifted left)."""

    label = "IA_SHIFT_LEFT_G"
    description = "Shift the green color plane left by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_labels: List[str] = ["IA_SHIFT_LEFT_G"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "G", "left")


class ShiftColorPlaneLeftR(AbstractImageAdjustmentTransformer):
    """Shift the Red channel left by 50% (BGR -> R shifted left)."""

    label = "IA_SHIFT_LEFT_R"
    description = "Shift the red color plane left by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_labels: List[str] = ["IA_SHIFT_LEFT_R"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "R", "left")


class ShiftColorPlaneDownB(AbstractImageAdjustmentTransformer):
    """Shift the Blue channel down by 50% (BGR -> B shifted down)."""

    label = "IA_SHIFT_DOWN_B"
    description = "Shift the blue color plane down by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_labels: List[str] = ["IA_SHIFT_DOWN_B"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "B", "down")


class ShiftColorPlaneDownG(AbstractImageAdjustmentTransformer):
    """Shift the Green channel down by 50% (BGR -> G shifted down)."""

    label = "IA_SHIFT_DOWN_G"
    description = "Shift the green color plane down by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_labels: List[str] = ["IA_SHIFT_DOWN_G"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "G", "down")


class ShiftColorPlaneDownR(AbstractImageAdjustmentTransformer):
    """Shift the Red channel down by 50% (BGR -> R shifted down)."""

    label = "IA_SHIFT_DOWN_R"
    description = "Shift the red color plane down by 50% while keeping BGR format; other channels unchanged."
    reverse_transformer_labels: List[str] = ["IA_SHIFT_DOWN_R"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_channel(image, "R", "down")


class ShiftColorPlaneLeftAll(AbstractImageAdjustmentTransformer):
    """Shift all three color channels left by 50% (BGR -> all channels shifted left)."""

    label = "IA_SHIFT_LEFT_ALL"
    description = "Shift all color planes left by 50% while keeping BGR format."
    reverse_transformer_labels: List[str] = ["IA_SHIFT_LEFT_ALL"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_all(image, "left")


class ShiftColorPlaneDownAll(AbstractImageAdjustmentTransformer):
    """Shift all three color channels down by 50% (BGR -> all channels shifted down)."""

    label = "IA_SHIFT_DOWN_ALL"
    description = "Shift all color planes down by 50% while keeping BGR format."
    reverse_transformer_labels: List[str] = ["IA_SHIFT_DOWN_ALL"]

    def transform(self, image: ndarray) -> ndarray:
        return _shift_all(image, "down")
