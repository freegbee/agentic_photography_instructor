from typing import Union, Tuple

from stable_baselines3.common.type_aliases import Schedule

from training.stable_baselines.hyperparameter.dqn_model_hyperparams import DqnModelParams
from training.stable_baselines.models.dqn_model_variants import DqnModelVariant
from training.stable_baselines.models.learning_rate_schedules import linear_schedule


class DqnParamsBuilder:
    def __init__(self,
                 variant: DqnModelVariant,
                 buffer_size: int,
                 batch_size: int,
                 learning_rate: Union[float, Schedule]):
        
        if isinstance(learning_rate, (float, int)):
            lr_schedule = linear_schedule(learning_rate)
        else:
            lr_schedule = learning_rate

        self._params: DqnModelParams = {
            "dqn_model_variant": variant,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "learning_rate": lr_schedule,
            # Defaults
            "learning_starts": 1000,
            "target_update_interval": 1000,
            "train_freq": 4,
            "gradient_steps": 1,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "gamma": 0.99
        }

    def with_exploration(self, fraction: float, initial_eps: float, final_eps: float) -> "DqnParamsBuilder":
        self._params["exploration_fraction"] = fraction
        self._params["exploration_initial_eps"] = initial_eps
        self._params["exploration_final_eps"] = final_eps
        return self

    def with_training_schedule(self,
                               learning_starts: int,
                               train_freq: int,
                               target_update_interval: int,
                               gradient_steps: int = 1) -> "DqnParamsBuilder":
        self._params["learning_starts"] = learning_starts
        self._params["train_freq"] = train_freq
        self._params["target_update_interval"] = target_update_interval
        self._params["gradient_steps"] = gradient_steps
        return self

    def build(self) -> DqnModelParams:
        return self._params

    @staticmethod
    def calculate_buffer_size(image_size: Tuple[int, int], target_memory_mb: int, channels: int = 3) -> int:
        """
        Berechnet die maximale buffer_size basierend auf dem verfügbaren Speicher.

        :param image_size: (width, height) des Bildes.
        :param target_memory_mb: Ziel-Speicherverbrauch in Megabytes.
        :param channels: Anzahl der Farbkanäle (Standard: 3 für RGB).
        :return: Anzahl der Einträge (buffer_size).
        """
        width, height = image_size
        bytes_per_pixel = 1  # uint8

        # Speicherbedarf pro Transition:
        # - Observation (Bild)
        # - Next Observation (Bild)
        # - Actions, Rewards, Dones (vernachlässigbar klein im Vergleich zum Bild, ca. 20-100 Bytes)
        image_bytes = width * height * channels * bytes_per_pixel
        transition_bytes = image_bytes * 2  # Obs + Next Obs

        total_bytes = target_memory_mb * 1024 * 1024

        return int(total_bytes / transition_bytes)