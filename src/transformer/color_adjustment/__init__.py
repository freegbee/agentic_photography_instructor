from .GrayscaleTransformer import GrayscaleTransformer
from .GrayscaleAverageTransformer import GrayscaleAverageTransformer
from .SwapColorChannelTransformer import SwapColorChannelTransformerGB, SwapColorChannelTransformerRB, \
    SwapColorChannelTransformerRG
from .InvertColorChannelTransformer import InvertColorChannelTransformerB, InvertColorChannelTransformerG, \
    InvertColorChannelTransformerR, InvertColorChannelTransformerBG, InvertColorChannelTransformerBR, \
    InvertColorChannelTransformerGR, InvertColorChannelTransformerAll
from .saturation_transformer import (
    SaturationIncreaseTransformerWeak,
    SaturationIncreaseTransformerMedium,
    SaturationIncreaseTransformerStrong,
    SaturationDecreaseTransformerWeak,
    SaturationDecreaseTransformerMedium,
    SaturationDecreaseTransformerStrong,
    VibranceTransformer
)
from .temperature_transformer import (
    WarmthTransformer,
    CoolnessTransformer
)
from .white_balance_transformer import AutoWhiteBalanceTransformer

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
    "VibranceTransformer",
    "WarmthTransformer",
    "CoolnessTransformer",
    "AutoWhiteBalanceTransformer"
]

SENSIBLE_CA_TRANSFORMERS = [
    GrayscaleTransformer.label, # GrayscaleAverageTransformer.label, InvertColorChannelTransformerAll.label,
    SaturationIncreaseTransformerWeak.label, SaturationDecreaseTransformerWeak.label,
    VibranceTransformer.label,
    WarmthTransformer.label, CoolnessTransformer.label,
    AutoWhiteBalanceTransformer.label
]
