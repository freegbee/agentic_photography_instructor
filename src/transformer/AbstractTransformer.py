from abc import ABC

from numpy import ndarray

from transformer.TransformerTypes import TransformerTypeEnum


class AbstractTransformer(ABC):
    """
    Abstract base class for all transformers.
    Each transformer must specify its type, label, and description.
    It must also implement the transform method to process input data.
    The label must be globally unique
    """
    def __init__(self, transformer_type: TransformerTypeEnum, label: str, description: str):
        self.transformerType = transformer_type
        self.label = label
        self.description = description

    def transform(self, data: ndarray) -> ndarray:
        raise NotImplementedError("Transform method must be implemented by subclasses.")