from abc import ABC, abstractmethod

from transformer.TransformerTypes import TransformerTypeEnum
from utils.Registries import TRANSFORMER_REGISTRY
from class_registry.base import AutoRegister

from numpy import ndarray

class AbstractTransformer(AutoRegister(TRANSFORMER_REGISTRY), ABC):
    """
    Abstract base class for all transformers.
    Each transformer must specify its type, label, and description.
    It must also implement the transform method to process input data.
    The label must be globally unique
    """

    transformer_type: TransformerTypeEnum = None
    label: str = None
    description: str = None

    def __init__(self):
        self.transformerType = self.__class__.transformer_type
        self.label = self.__class__.label
        self.description  = self.__class__.description

    @abstractmethod
    def transform(self, data: ndarray) -> ndarray:
        pass
