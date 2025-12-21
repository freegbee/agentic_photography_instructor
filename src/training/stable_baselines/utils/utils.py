from typing import List

from transformer.AbstractTransformer import AbstractTransformer
from utils.Registries import TRANSFORMER_REGISTRY


def get_consistent_transformers(transformer_labels: List[str]) -> List[AbstractTransformer]:
    return [TRANSFORMER_REGISTRY.get(name) for name in sorted(transformer_labels)]