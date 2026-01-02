import logging
from pathlib import Path
from typing import Any, SupportsFloat, List, Tuple

from gymnasium.core import ObsType
from gymnasium.spaces import Discrete
from numpy import integer

from data_types.AgenticImage import ImageData
from juror_client import JurorClient
from training.stable_baselines.environment.image_transform_env import ImageTransformEnv
from training.stable_baselines.environment.samplers import CocoDatasetSampler
from transformer.AbstractTransformer import AbstractTransformer

logger = logging.getLogger(__name__)


class ImageOptimizationEnv(ImageTransformEnv):
    """
    Eine Variante des ImageTransformEnv mit angepasster Logik (z.B. Reward-Shaping oder Observation).
    """

    def __init__(self,
                 transformers: List[AbstractTransformer],
                 coco_dataset_sampler: CocoDatasetSampler,
                 juror_client: JurorClient,
                 success_bonus: float,
                 image_max_size: Tuple[int, int],
                 max_transformations: int = 5,
                 max_action_param_dim: int = 1,
                 seed: int = 42,
                 render_mode: str = "imshow",  # | "save"
                 render_save_dir: Path = None):
        super().__init__(transformers,
                         coco_dataset_sampler,
                         juror_client,
                         success_bonus,
                         image_max_size,
                         max_transformations,
                         max_action_param_dim,
                         seed,
                         render_mode,
                         render_save_dir)

        # weitere Initialisierung
        self.stop_bonus: float = 0.0        # Bonus, wenn Agent "STOP" findet

    def _calculate_action_space(self) -> Discrete[integer[Any] | Any]:
        return Discrete(len(self.transformers) + 1)  # Die Stop-Action kommt noch dazu

    def _reset_image_properties(self, image_data: ImageData, random_index: int):
        self.current_image = image_data.image_data
        self.current_image_id = image_data.id
        self.initial_score = image_data.initial_score  # hier ist der initial score unser startpunkt für die Optimierung
        self.current_score = image_data.initial_score  # und dies ist dann auch der aktuelle score

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_val = int(action)
        previous_score = self.current_score

        terminated = False
        truncated = False

        if self._is_stop_action(action_val):
            reward = self._calculate_stop_reward(previous_score)
            terminated = True
        else:
            self.current_image, self.current_score = self._transform_and_score(self.current_image,
                                                                               self.transformers[action_val])
            reward = self._calculate_reward(previous_score, self.current_score)

        self.step_count += 1
        
        # Truncated wenn Max Steps erreicht UND nicht explizit gestoppt wurde
        if self.step_count >= self.max_transformations and not terminated:
            truncated = True

        info = {
            "score": self.current_score,
            "steps": self.step_count,
            "success": self.current_score > self.initial_score,
            "initial_score": self.initial_score
        }

        return self.current_image, reward, terminated, truncated, info

    def _is_stop_action(self, action: int) -> bool:
        return action == len(self.transformers)  # letzte action ist die stop action

    def _is_success(self) -> bool:
        return False

    def _calculate_reward(self, previous_score: float, new_score: float):
        """
        Potential-Based Reward Shaping mit einer konvexen Funktion (x^2).
        Dies belohnt Verbesserungen bei bereits hohen Scores stärker als bei niedrigen Scores.
        
        Beispiel:
        2.0 -> 2.1: Delta 0.1, Reward = 2.1^2 - 2.0^2 = 0.41
        8.0 -> 8.1: Delta 0.1, Reward = 8.1^2 - 8.0^2 = 1.61
        """
        return (new_score ** 2) - (previous_score ** 2)

    def _calculate_stop_reward(self, previous_score: float) -> float:
        """
        Berechnet den Reward, falls STOP erreicht wird.
        Denkbar wäre z.B. etwas zu belohnen, wenn früher gestoppt wird, aber nur wenn, der Score besser wurde
        """
        return self.stop_bonus
