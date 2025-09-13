from abc import abstractmethod

from numpy import ndarray

from transformer.AbstractTransformer import AbstractTransformer
from transformer.TransformerTypes import TransformerTypeEnum


class AbstractColorAdjustmentTransformer(AbstractTransformer):
    transformer_type = TransformerTypeEnum.COLOR_ADJUSTMENT

    @abstractmethod
    def transform(self, data: ndarray) -> ndarray:
        pass
