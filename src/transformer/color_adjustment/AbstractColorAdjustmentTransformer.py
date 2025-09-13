from transformer.AbstractTransformer import AbstractTransformer
from transformer.TransformerTypes import TransformerTypeEnum


class AbstractColorAdjustmentTransformer(AbstractTransformer):
    def __init__(self, label: str, description: str):
        super().__init__(TransformerTypeEnum.COLOR_ADJUSTMENT, label, description)