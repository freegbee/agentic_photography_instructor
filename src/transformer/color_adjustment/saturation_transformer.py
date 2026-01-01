from abc import abstractmethod
import cv2
from numpy import ndarray

from transformer.color_adjustment.AbstractColorAdjustmentTransformer import AbstractColorAdjustmentTransformer


class AbstractSaturationTransformer(AbstractColorAdjustmentTransformer):
    """Base class for saturation adjustments using HSV color space."""

    @property
    @abstractmethod
    def saturation_factor(self) -> float:
        pass

    def transform(self, image: ndarray) -> ndarray:
        # Convert BGR to HSV to separate color information from intensity
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Split channels: Hue, Saturation, Value
        h, s, v = cv2.split(hsv)
        
        # Apply scaling to the Saturation channel
        # cv2.convertScaleAbs computes: alpha * src + beta
        # It handles clipping to 0-255 and casting to uint8 automatically.
        s_new = cv2.convertScaleAbs(s, alpha=self.saturation_factor, beta=0)
        
        # Merge channels back and convert to BGR
        hsv_new = cv2.merge([h, s_new, v])
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)


class SaturationIncreaseTransformerWeak(AbstractSaturationTransformer):
    label = "CA_SAT_INC_WEAK"
    description = "Increase saturation weakly (factor=1.2)."
    saturation_factor = 1.2
    reverse_transformer_label = "CA_SAT_DEC_WEAK"


class SaturationIncreaseTransformerMedium(AbstractSaturationTransformer):
    label = "CA_SAT_INC_MED"
    description = "Increase saturation medium (factor=1.5)."
    saturation_factor = 1.5
    reverse_transformer_label = "CA_SAT_DEC_MED"


class SaturationIncreaseTransformerStrong(AbstractSaturationTransformer):
    label = "CA_SAT_INC_STRONG"
    description = "Increase saturation strongly (factor=2.0)."
    saturation_factor = 2.0
    reverse_transformer_label = "CA_SAT_DEC_STRONG"


class SaturationDecreaseTransformerWeak(AbstractSaturationTransformer):
    label = "CA_SAT_DEC_WEAK"
    description = "Decrease saturation weakly (factor=0.8)."
    saturation_factor = 0.8
    reverse_transformer_label = "CA_SAT_INC_WEAK"


class SaturationDecreaseTransformerMedium(AbstractSaturationTransformer):
    label = "CA_SAT_DEC_MED"
    description = "Decrease saturation medium (factor=0.6)."
    saturation_factor = 0.6
    reverse_transformer_label = "CA_SAT_INC_MED"


class SaturationDecreaseTransformerStrong(AbstractSaturationTransformer):
    label = "CA_SAT_DEC_STRONG"
    description = "Decrease saturation strongly (factor=0.4)."
    saturation_factor = 0.4
    reverse_transformer_label = "CA_SAT_INC_STRONG"