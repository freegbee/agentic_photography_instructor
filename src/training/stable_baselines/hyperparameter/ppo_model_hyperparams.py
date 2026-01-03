from typing import TypedDict

from stable_baselines3.common.type_aliases import Schedule
from training.stable_baselines.models.model_variants import PpoModelVariant


class PpoModelParams(TypedDict):
    ppo_model_variant: PpoModelVariant
    n_steps: int
    batch_size: int
    n_epochs: int
    model_learning_schedule: Schedule