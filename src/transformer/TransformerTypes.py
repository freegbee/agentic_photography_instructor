from enum import Enum


class TransformerTypeEnum(str, Enum):
    CROP = "Crop"
    COLOR_ADJUSTMENT = "ColorAdjustment"
    IMAGE_ADJUSTMENT = "ImageAdjustment"
    ROTATION = "Rotation"
    FLIP = "Flip"