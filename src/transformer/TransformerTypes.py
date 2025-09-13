from enum import Enum


class TransformerTypeEnum(str, Enum):
    CROP = "Crop"
    COLOR_ADJUSTMENT = "ColorAdjustment"
    ROTATION = "Rotation"
    FLIP = "Flip"