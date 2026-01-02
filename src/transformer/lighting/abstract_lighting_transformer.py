from abc import abstractmethod

from numpy import ndarray

from transformer.AbstractTransformer import AbstractTransformer
from transformer.TransformerTypes import TransformerTypeEnum


class AbstractLightingTransformer(AbstractTransformer):
    """
    Abstract class for image adjustments like blur, sharpen etc.

    See https://www.geeksforgeeks.org/python/image-filtering-using-convolution-in-opencv/ for inspiration.
    """
    transformer_type = TransformerTypeEnum.LIGHTING

    @abstractmethod
    def transform(self, data: ndarray) -> ndarray:
        pass
