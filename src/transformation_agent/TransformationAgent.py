from typing import List, Tuple

from numpy import ndarray

from transformer.AbstractTransformer import AbstractTransformer
from utils.Registries import TRANSFORMER_REGISTRY


class TransformationAgent:

    def __init__(self, transformer_labels: List[str]):
        self.transformer_labels: List[str] = transformer_labels

    def transform(self, image: ndarray) -> Tuple[ndarray, str]:
        transformation_labels: List[str] = []
        for tl in self.transformer_labels:
            t: AbstractTransformer = TRANSFORMER_REGISTRY.get(tl)
            image = t.transform(image)
            transformation_labels.append(t.label)

        return image, '&'.join(transformation_labels)
