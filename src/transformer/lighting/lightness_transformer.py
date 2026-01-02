from abc import abstractmethod
import cv2
from numpy import ndarray

from transformer.lighting.abstract_lighting_transformer import AbstractLightingTransformer


class AbstractLightnessTransformer(AbstractLightingTransformer):
    """Base class for brightness (lightness) adjustments."""

    @property
    @abstractmethod
    def beta(self) -> float:
        pass

    def transform(self, image: ndarray) -> ndarray:
        # cv2.convertScaleAbs calculates: alpha * src + beta
        # We use alpha=1.0 to only change brightness (offset).
        return cv2.convertScaleAbs(image, alpha=1.0, beta=self.beta)


class LightnessIncreaseTransformerWeak(AbstractLightnessTransformer):
    label = "LI_LIGHTNESS_INC_WEAK"
    description = "Increase brightness weakly (beta=20)."
    beta = 20.0
    reverse_transformer_label = "LI_LIGHTNESS_DEC_WEAK"


class LightnessIncreaseTransformerMedium(AbstractLightnessTransformer):
    label = "LI_LIGHTNESS_INC_MED"
    description = "Increase brightness medium (beta=40)."
    beta = 40.0
    reverse_transformer_label = "LI_LIGHTNESS_DEC_MED"


class LightnessIncreaseTransformerStrong(AbstractLightnessTransformer):
    label = "LI_LIGHTNESS_INC_STRONG"
    description = "Increase brightness strongly (beta=60)."
    beta = 60.0
    reverse_transformer_label = "LI_LIGHTNESS_DEC_STRONG"


class LightnessDecreaseTransformerWeak(AbstractLightnessTransformer):
    label = "LI_LIGHTNESS_DEC_WEAK"
    description = "Decrease brightness weakly (beta=-20)."
    beta = -20.0
    reverse_transformer_label = "LI_LIGHTNESS_INC_WEAK"


class LightnessDecreaseTransformerMedium(AbstractLightnessTransformer):
    label = "LI_LIGHTNESS_DEC_MED"
    description = "Decrease brightness medium (beta=-40)."
    beta = -40.0
    reverse_transformer_label = "LI_LIGHTNESS_INC_MED"


class LightnessDecreaseTransformerStrong(AbstractLightnessTransformer):
    label = "LI_LIGHTNESS_DEC_STRONG"
    description = "Decrease brightness strongly (beta=-60)."
    beta = -60.0
    reverse_transformer_label = "LI_LIGHTNESS_INC_STRONG"