from .center_square_crop_transformer import CenterSquareCropTransformer
from .rule_of_thirds_crop_transformer import RuleOfThirdsCropTransformer
from .aspect_ratio_crop_transformer import (
    AspectRatio16x9CropTransformer,
    AspectRatio4x3CropTransformer,
    AspectRatio3x2CropTransformer,
    AspectRatio4x5CropTransformer,
    AspectRatio5x4CropTransformer,
    AspectRatio1x1CropTransformer
)
from .horizon_leveling_transformer import HorizonLevelingTransformer

__all__ = [
    "CenterSquareCropTransformer",
    "RuleOfThirdsCropTransformer",
    "AspectRatio16x9CropTransformer",
    "AspectRatio4x3CropTransformer",
    "AspectRatio3x2CropTransformer",
    "AspectRatio4x5CropTransformer",
    "AspectRatio5x4CropTransformer",
    "AspectRatio1x1CropTransformer",
    "HorizonLevelingTransformer",
]

SENSIBLE_CROP_TRANSFORMERS = [
    RuleOfThirdsCropTransformer.label,
    AspectRatio16x9CropTransformer.label,
    AspectRatio4x3CropTransformer.label,
    AspectRatio3x2CropTransformer.label,
    AspectRatio4x5CropTransformer.label,
    AspectRatio5x4CropTransformer.label,
    AspectRatio1x1CropTransformer.label,
    HorizonLevelingTransformer.label,
]