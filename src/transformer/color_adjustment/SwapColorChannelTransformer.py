from typing import Literal

from numpy import ndarray

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import (
    AbstractColorAdjustmentTransformer,
)


Channel = Literal["B", "G", "R"]


def _swap_channels(image: ndarray, channel_a: Channel, channel_b: Channel) -> ndarray:
    """
    Return a copy of the image with two color channels swapped.
    - The project uses BGR channel order (OpenCV conventions).
    - Keeps 3 channels and preserves channel order semantics overall (just swaps two positions).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an HxWx3 BGR ndarray")

    color_channel_indices = {"B": 0, "G": 1, "R": 2}
    channel_index_a, channel_index_b = color_channel_indices[channel_a], color_channel_indices[channel_b]

    # Work on a copy to avoid mutating the input unexpectedly
    out = image.copy()
    out[..., channel_index_a] = image[..., channel_index_b]
    out[..., channel_index_b] = image[..., channel_index_a]
    return out


class SwapColorChannelTransformerRG(AbstractColorAdjustmentTransformer):
    """Swap Red and Green channels (BGR -> BGR with R and G swapped)."""

    label = "CA_SWAP_RG"
    description = "Swap the red and green color channels while keeping BGR format."

    def transform(self, image: ndarray) -> ndarray:
        return _swap_channels(image, "R", "G")


class SwapColorChannelTransformerRB(AbstractColorAdjustmentTransformer):
    """Swap Red and Blue channels (BGR -> BGR with R and B swapped)."""

    label = "CA_SWAP_RB"
    description = "Swap the red and blue color channels while keeping BGR format."

    def transform(self, image: ndarray) -> ndarray:
        return _swap_channels(image, "R", "B")


class SwapColorChannelTransformerGB(AbstractColorAdjustmentTransformer):
    """Swap Green and Blue channels (BGR -> BGR with G and B swapped)."""

    label = "CA_SWAP_GB"
    description = "Swap the green and blue color channels while keeping BGR format."

    def transform(self, image: ndarray) -> ndarray:
        return _swap_channels(image, "G", "B")
