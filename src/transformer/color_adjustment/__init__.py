from .GrayscaleTransformer import GrayscaleTransformer
from .GrayscaleAverageTransformer import GrayscaleAverageTransformer
from .SwapColorChannelTransformer import SwapColorChannelTransformerGB, SwapColorChannelTransformerRB, SwapColorChannelTransformerRG
from .InvertColorChannelTransformer import InvertColorChannelTransformerB, InvertColorChannelTransformerG, InvertColorChannelTransformerR, InvertColorChannelTransformerBG, InvertColorChannelTransformerBR, InvertColorChannelTransformerGR, InvertColorChannelTransformerAll


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
]

SENSIBLE_CA_TRANSFORMERS = [GrayscaleTransformer.label, GrayscaleAverageTransformer.label, InvertColorChannelTransformerAll.label]