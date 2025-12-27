from abc import ABC, abstractmethod
from typing import Union

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
        """
        Apply the transformation to the input data. The input data must be a numpy ndarray witht 3 channels.
        this may not be changed. The color channel order may not be changed.
        :param data: image data as numpy ndarray
        :return: transformed image data
        """
        pass

    def get_reverse_transformer_label(self) -> Union[str, None]:
        """ Returns the label(s) of the transformer(s) that can reverse this transformation.
        If no reverse transformer is defined, returns an empty list.
        """
        if hasattr(self.__class__, 'reverse_transformer_label'):
            return self.__class__.reverse_transformer_label
        else:
            return None
