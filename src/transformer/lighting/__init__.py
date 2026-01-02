from .contrast_transformer import (
    ContrastIncreaseTransformerWeak,
    ContrastIncreaseTransformerMedium,
    ContrastIncreaseTransformerStrong,
    ContrastDecreaseTransformerWeak,
    ContrastDecreaseTransformerMedium,
    ContrastDecreaseTransformerStrong,
    CLAHETransformer
)
from .lightness_transformer import (
    LightnessIncreaseTransformerWeak,
    LightnessIncreaseTransformerMedium,
    LightnessIncreaseTransformerStrong,
    LightnessDecreaseTransformerWeak,
    LightnessDecreaseTransformerMedium,
    LightnessDecreaseTransformerStrong
)
from .gamma_transformer import AutoGammaCorrectionTransformer

__all__ = [
    "ContrastIncreaseTransformerWeak",
    "ContrastIncreaseTransformerMedium",
    "ContrastIncreaseTransformerStrong",
    "ContrastDecreaseTransformerWeak",
    "ContrastDecreaseTransformerMedium",
    "ContrastDecreaseTransformerStrong",
    "CLAHETransformer",
    "LightnessIncreaseTransformerWeak",
    "LightnessIncreaseTransformerMedium",
    "LightnessIncreaseTransformerStrong",
    "LightnessDecreaseTransformerWeak",
    "LightnessDecreaseTransformerMedium",
    "LightnessDecreaseTransformerStrong",
    "AutoGammaCorrectionTransformer"
]

SENSIBLE_LIGHTING_TRANSFORMERS = [
    ContrastIncreaseTransformerWeak.label, ContrastDecreaseTransformerWeak.label,
    LightnessIncreaseTransformerWeak.label, LightnessDecreaseTransformerWeak.label,
    AutoGammaCorrectionTransformer.label,
    CLAHETransformer.label
]