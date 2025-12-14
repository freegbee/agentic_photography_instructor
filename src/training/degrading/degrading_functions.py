from abc import ABC, abstractmethod
from typing import List, Tuple

from numpy import ndarray

from training.hyperparameter_registry import HyperparameterRegistry
from training.rl_training.training_params import GeneralPreprocessingParams, TransformPreprocessingParams
from transformer.AbstractTransformer import AbstractTransformer
from utils.Registries import TRANSFORMER_REGISTRY


class AbstractImageDegradingFunction(ABC):

    def __init__(self, name: str):
        self.name = name

    def degrade(self, image: ndarray) -> Tuple[ndarray, List[str]]:
        applied_transformers: List[str]  = []
        for transformer in self._get_transformers():
            image = transformer.transform(image)
            applied_transformers.append(transformer.label)
        return image, applied_transformers

    @abstractmethod
    def _get_transformers(self) -> List[AbstractTransformer]:
        raise NotImplementedError()


class SingleTransformationDegradingFunction(AbstractImageDegradingFunction):

    def __init__(self, transformer_label):
        super().__init__("SingleTransformationDegradingFunction")
        self.transformer: AbstractTransformer = TRANSFORMER_REGISTRY.get(transformer_label)

    def __str__(self):
        return f"SingleTransformationDegradingFunction({self.transformer.label})"

    def __repr__(self):
        return f"SingleTransformationDegradingFunction({self.transformer.label})"

    def _get_transformers(self) -> List[AbstractTransformer]:
        return [self.transformer]


class RandomTransformationDegradingFunction(AbstractImageDegradingFunction):

    def __init__(self, transformer_labels: List[str]):
        super().__init__("RandomTransformationDegradingFunction")
        import random
        general_params = HyperparameterRegistry.get_store(GeneralPreprocessingParams).get()
        self.random_generator = random.Random(general_params["random_seed"])
        self.transformers: List[AbstractTransformer] = [TRANSFORMER_REGISTRY.get(name) for name in transformer_labels]

    def __str__(self):
        transformer_labels = ', '.join([transformer.label for transformer in self.transformers])
        return f"RandomTransformationDegradingFunction([{transformer_labels}])"

    def __repr__(self):
        transformer_labels = ', '.join([transformer.label for transformer in self.transformers])
        return f"RandomTransformationDegradingFunction([{transformer_labels}])"

    def _get_transformers(self) -> List[AbstractTransformer]:
        return [self.random_generator.choice(self.transformers)]


class SequentialTransformationDegradingFunction(AbstractImageDegradingFunction):

    def __init__(self, transformer_labels: List[str]):
        super().__init__("SequentialTransformationDegradingFunction")
        self.transformers: List[AbstractTransformer] = [TRANSFORMER_REGISTRY.get(name) for name in transformer_labels]
        self._idx: int = 0

    def __str__(self):
        transformer_labels = ', '.join([transformer.label for transformer in self.transformers])
        return f"SequentialTransformationDegradingFunction([{transformer_labels}])"

    def __repr__(self):
        transformer_labels = ', '.join([transformer.label for transformer in self.transformers])
        return f"SequentialTransformationDegradingFunction([{transformer_labels}])"

    def _get_transformers(self) -> List[AbstractTransformer]:
        current = self._idx
        # Index zyklisch erhöhen
        self._idx = (self._idx + 1) % len(self.transformers)
        return [self.transformers[current]]


class DegradingFunctionFactory:

    @staticmethod
    def create_degrading_function(use_random: bool, transformer_labels: List[str]) -> AbstractImageDegradingFunction:
        if use_random:
            return RandomTransformationDegradingFunction(transformer_labels)
        elif len(transformer_labels) == 1:
            return SingleTransformationDegradingFunction(transformer_labels[0])
        else:
            return SequentialTransformationDegradingFunction(transformer_labels)

    @staticmethod
    def create_from_hyperparams() -> AbstractImageDegradingFunction:
        general_params = HyperparameterRegistry.get_store(TransformPreprocessingParams).get()
        return DegradingFunctionFactory.create_degrading_function(
            use_random=general_params["use_random_transformer"],
            transformer_labels=general_params["transformer_names"]
        )