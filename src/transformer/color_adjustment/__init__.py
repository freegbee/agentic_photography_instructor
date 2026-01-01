from .GrayscaleTransformer import GrayscaleTransformer
from .GrayscaleAverageTransformer import GrayscaleAverageTransformer
from .SwapColorChannelTransformer import SwapColorChannelTransformerGB, SwapColorChannelTransformerRB, SwapColorChannelTransformerRG
from .InvertColorChannelTransformer import InvertColorChannelTransformerB, InvertColorChannelTransformerG, InvertColorChannelTransformerR, InvertColorChannelTransformerBG, InvertColorChannelTransformerBR, InvertColorChannelTransformerGR, InvertColorChannelTransformerAll
from .saturation_transformer import (
    SaturationIncreaseTransformerWeak,
    SaturationIncreaseTransformerMedium,
    SaturationIncreaseTransformerStrong,
    SaturationDecreaseTransformerWeak,
    SaturationDecreaseTransformerMedium,
    SaturationDecreaseTransformerStrong,
    VibranceTransformer
)


__all__ = [
    "GrayscaleTransformer",
    "GrayscaleAverageTransformer",
    "SwapColorChannelTransformerGB",
    "SwapColorChannelTransformerRB",
    "SwapColorChannelTransformerRG",
    "InvertColorChannelTransformerB",
    "InvertColorChannelTransformerG",
    "InvertColorChannelTransformerR",
    "InvertColorChannelTransformerBG",
    "InvertColorChannelTransformerBR",
    "InvertColorChannelTransformerGR",
    "InvertColorChannelTransformerAll",
    "SaturationIncreaseTransformerWeak",
    "SaturationIncreaseTransformerMedium",
    "SaturationIncreaseTransformerStrong",
    "SaturationDecreaseTransformerWeak",
    "SaturationDecreaseTransformerMedium",
    "SaturationDecreaseTransformerStrong",
    "VibranceTransformer"
]

SENSIBLE_CA_TRANSFORMERS = [
    GrayscaleTransformer.label, GrayscaleAverageTransformer.label, InvertColorChannelTransformerAll.label,
    SaturationIncreaseTransformerWeak.label, SaturationDecreaseTransformerWeak.label,
    VibranceTransformer.label
]