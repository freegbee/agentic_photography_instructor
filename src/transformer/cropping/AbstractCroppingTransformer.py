from abc import abstractmethod

from numpy import ndarray

from transformer.AbstractTransformer import AbstractTransformer
from transformer.TransformerTypes import TransformerTypeEnum


class AbstractCroppingTransformer(AbstractTransformer):
    """
    Abstract class for cropping transformers.

    See https://opencv.org/blog/cropping-an-image-using-opencv/ for inspiration.
    """

    transformer_type = TransformerTypeEnum.CROP

    @abstractmethod
    def transform(self, data: ndarray) -> ndarray:
        pass
