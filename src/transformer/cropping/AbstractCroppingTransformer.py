from transformer.AbstractTransformer import AbstractTransformer
from transformer.TransformerTypes import TransformerTypeEnum


class AbstractCroppingTransformer(AbstractTransformer):
    """
    Abstract class for cropping transformers.

    See https://opencv.org/blog/cropping-an-image-using-opencv/ for inspiration.
    """
    def __init__(self, label: str, description: str):
        super().__init__(TransformerTypeEnum.CROP, label, description)