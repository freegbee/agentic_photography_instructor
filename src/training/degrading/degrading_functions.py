from abc import ABC, abstractmethod
from typing import List, Tuple, Set
import random
import logging

from numpy import ndarray

from training.hyperparameter_registry import HyperparameterRegistry
from training.rl_training.training_params import GeneralPreprocessingParams, TransformPreprocessingParams
from transformer.AbstractTransformer import AbstractTransformer
from transformer.TransformationComposition import get_composition_table
from utils.Registries import TRANSFORMER_REGISTRY

logger = logging.getLogger(__name__)


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


class MultiRandomTransformationDegradingFunction(AbstractImageDegradingFunction):
    """Applies N random transformations to an image without duplicates or reversing pairs."""

    def __init__(self, transformer_labels: List[str], num_transformations: int = 2):
        super().__init__("MultiRandomTransformationDegradingFunction")
        general_params = HyperparameterRegistry.get_store(GeneralPreprocessingParams).get()
        self.random_generator = random.Random(general_params["random_seed"])
        self.transformer_labels = transformer_labels
        self.num_transformations = num_transformations

        # Build reverse transformer mapping
        self.reverse_map: dict[str, Set[str]] = {}
        for label in transformer_labels:
            transformer = TRANSFORMER_REGISTRY.get(label)
            reverse_label = transformer.get_reverse_transformer_label()
            if reverse_label:
                if isinstance(reverse_label, list):
                    self.reverse_map[label] = set(reverse_label)
                else:
                    self.reverse_map[label] = {reverse_label}
            else:
                self.reverse_map[label] = set()

    def __str__(self):
        return f"MultiRandomTransformationDegradingFunction({self.num_transformations} from {len(self.transformer_labels)} transformers)"

    def __repr__(self):
        return f"MultiRandomTransformationDegradingFunction(num_transformations={self.num_transformations}, transformer_labels={self.transformer_labels})"

    def _get_transformers(self) -> List[AbstractTransformer]:
        """Select N random transformations without duplicates, reversing pairs, or composable sequences."""
        composition_table = get_composition_table()
        selected_labels: List[str] = []
        excluded_labels: Set[str] = set()

        available_labels = list(self.transformer_labels)
        self.random_generator.shuffle(available_labels)

        for label in available_labels:
            if len(selected_labels) >= self.num_transformations:
                break

            # Skip if already selected or excluded
            if label in excluded_labels:
                continue

            # Check if this would create a reducible composition with any previously selected transformer
            creates_composition = False
            for selected_label in selected_labels:
                if composition_table.is_pair_blocked(selected_label, label):
                    logger.debug(f"Skipping {label} because it composes with {selected_label}")
                    creates_composition = True
                    break

            if creates_composition:
                continue

            selected_labels.append(label)
            excluded_labels.add(label)

            # Exclude reverse transformers
            if label in self.reverse_map:
                excluded_labels.update(self.reverse_map[label])

        if len(selected_labels) < self.num_transformations:
            raise ValueError(f"Could not select {self.num_transformations} valid transformations. Only {len(selected_labels)} available without conflicts.")

        return [TRANSFORMER_REGISTRY.get(label) for label in selected_labels]


class DegradingFunctionFactory:

    @staticmethod
    def create_degrading_function(use_random: bool, transformer_labels: List[str], num_transformations: int = 1) -> AbstractImageDegradingFunction:
        if num_transformations > 1:
            return MultiRandomTransformationDegradingFunction(transformer_labels, num_transformations)
        elif use_random:
            return RandomTransformationDegradingFunction(transformer_labels)
        elif len(transformer_labels) == 1:
            return SingleTransformationDegradingFunction(transformer_labels[0])
        else:
            return SequentialTransformationDegradingFunction(transformer_labels)

    @staticmethod
    def create_from_hyperparams() -> AbstractImageDegradingFunction:
        general_params = HyperparameterRegistry.get_store(TransformPreprocessingParams).get()
        num_transformations = general_params.get("num_transformations", 1)
        return DegradingFunctionFactory.create_degrading_function(
            use_random=general_params["use_random_transformer"],
            transformer_labels=general_params["transformer_names"],
            num_transformations=num_transformations
        )