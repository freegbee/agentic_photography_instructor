from .contrast_transformer import (
    ContrastIncreaseTransformerWeak,
    ContrastIncreaseTransformerMedium,
    ContrastIncreaseTransformerStrong,
    ContrastDecreaseTransformerWeak,
    ContrastDecreaseTransformerMedium,
    ContrastDecreaseTransformerStrong,
    CLAHETransformer,
    AutoContrastTransformer
)
from .lightness_transformer import (
    LightnessIncreaseTransformerWeak,
    LightnessIncreaseTransformerMedium,
    LightnessIncreaseTransformerStrong,
    LightnessDecreaseTransformerWeak,
    LightnessDecreaseTransformerMedium,
    LightnessDecreaseTransformerStrong
)
from .gamma_transformer import (
    AutoGammaCorrectionTransformer,
    GammaBrightenTransformer,
    GammaDarkenTransformer
)

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
    "AutoGammaCorrectionTransformer",
    "GammaBrightenTransformer",
    "GammaDarkenTransformer",
    "AutoContrastTransformer"
]

SENSIBLE_LIGHTING_TRANSFORMERS = [
    ContrastIncreaseTransformerWeak.label, ContrastDecreaseTransformerWeak.label,
    LightnessIncreaseTransformerWeak.label, LightnessDecreaseTransformerWeak.label,
    AutoGammaCorrectionTransformer.label,
    GammaBrightenTransformer.label, GammaDarkenTransformer.label,
    CLAHETransformer.label,
    AutoContrastTransformer.label
]
