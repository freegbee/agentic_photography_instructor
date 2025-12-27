from typing import Iterable, Literal, Set, List

import numpy as np
from numpy import ndarray

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import (
    AbstractColorAdjustmentTransformer,
)

Channel = Literal["B", "G", "R"]


def _invert_channels(image: ndarray, channels: Iterable[Channel]) -> ndarray:
    """
    Return a copy of the image with the specified color channels inverted.
    - Assumes BGR channel order (OpenCV conventions).
    - Keeps 3 channels and preserves channel order; only per-channel values are inverted.
    - For uint8 images, inversion is computed as 255 - value.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an HxWx3 BGR ndarray")

    channel_set: Set[Channel] = set(channels)
    if not channel_set:
        return image

    out = image.copy()

    # Use uint8 convention (most common). If not uint8, attempt a reasonable default.
    if out.dtype == np.uint8:
        max_val = 255
    elif np.issubdtype(out.dtype, np.integer):
        # Infer max from dtype (e.g., uint16)
        info = np.iinfo(out.dtype)
        max_val = info.max
    else:
        # Float images: assume range 0..1 if values <= 1.5, else 0..255
        max_val = 1.0 if float(out.max(initial=0)) <= 1.5 else 255.0

    idx_map = {"B": 0, "G": 1, "R": 2}
    for ch in channel_set:
        ch_idx = idx_map[ch]
        out[..., ch_idx] = max_val - out[..., ch_idx]

    return out


class InvertColorChannelTransformerB(AbstractColorAdjustmentTransformer):
    """Invert the Blue channel (BGR -> B inverted, G/R unchanged)."""

    label = "CA_INV_B"
    description = "Invert the blue color channel while keeping BGR format and other channels unchanged."
    reverse_transformer_label = "CA_INV_B"

    def transform(self, image: ndarray) -> ndarray:
        return _invert_channels(image, ["B"])


class InvertColorChannelTransformerG(AbstractColorAdjustmentTransformer):
    """Invert the Green channel (BGR -> G inverted, B/R unchanged)."""

    label = "CA_INV_G"
    description = "Invert the green color channel while keeping BGR format and other channels unchanged."
    reverse_transformer_label = "CA_INV_G"

    def transform(self, image: ndarray) -> ndarray:
        return _invert_channels(image, ["G"])


class InvertColorChannelTransformerR(AbstractColorAdjustmentTransformer):
    """Invert the Red channel (BGR -> R inverted, B/G unchanged)."""

    label = "CA_INV_R"
    description = "Invert the red color channel while keeping BGR format and other channels unchanged."
    reverse_transformer_label = "CA_INV_R"

    def transform(self, image: ndarray) -> ndarray:
        return _invert_channels(image, ["R"])


class InvertColorChannelTransformerBG(AbstractColorAdjustmentTransformer):
    """Invert Blue and Green channels (BGR -> B,G inverted; R unchanged)."""

    label = "CA_INV_BG"
    description = "Invert the blue and green color channels while keeping BGR format; red unchanged."
    reverse_transformer_label = "CA_INV_BG"

    def transform(self, image: ndarray) -> ndarray:
        return _invert_channels(image, ["B", "G"])


class InvertColorChannelTransformerBR(AbstractColorAdjustmentTransformer):
    """Invert Blue and Red channels (BGR -> B,R inverted; G unchanged)."""

    label = "CA_INV_BR"
    description = "Invert the blue and red color channels while keeping BGR format; green unchanged."
    reverse_transformer_label = "CA_INV_BR"

    def transform(self, image: ndarray) -> ndarray:
        return _invert_channels(image, ["B", "R"])


class InvertColorChannelTransformerGR(AbstractColorAdjustmentTransformer):
    """Invert Green and Red channels (BGR -> G,R inverted; B unchanged)."""

    label = "CA_INV_GR"
    description = "Invert the green and red color channels while keeping BGR format; blue unchanged."
    reverse_transformer_label = "CA_INV_GR"

    def transform(self, image: ndarray) -> ndarray:
        return _invert_channels(image, ["G", "R"])


class InvertColorChannelTransformerAll(AbstractColorAdjustmentTransformer):
    """Invert all channels (BGR -> all inverted)."""

    label = "CA_INV_ALL"
    description = "Invert all B, G, and R color channels while keeping BGR format."
    reverse_transformer_label = "CA_INV_ALL"

    def transform(self, image: ndarray) -> ndarray:
        return _invert_channels(image, ["B", "G", "R"])