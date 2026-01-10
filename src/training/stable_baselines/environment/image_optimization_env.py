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
from training.stable_baselines.rewards.reward_strategies import AbstractRewardStrategy
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
                 reward_strategy: AbstractRewardStrategy,
                 image_max_size: Tuple[int, int],
                 max_transformations: int = 5,
                 max_action_param_dim: int = 1,
                 seed: int = 42,
                 render_mode: str = "imshow",  # | "save"
                 render_save_dir: Path = None):
        super().__init__(transformers,
                         coco_dataset_sampler,
                         juror_client,
                         reward_strategy,
                         image_max_size,
                         max_transformations,
                         max_action_param_dim,
                         seed,
                         render_mode,
                         render_save_dir)

        # weitere Initialisierung
        self.stop_bonus: float = 0.0        # Bonus, wenn Agent "STOP" findet
        self.has_score_decreased = False
        self.mdp_active = False

    def _calculate_action_space(self) -> Discrete[integer[Any] | Any]:
        return Discrete(len(self.transformers) + 1)  # Die Stop-Action kommt noch dazu

    def _reset_image_properties(self, image_data: ImageData, random_index: int):
        self.current_image = image_data.image_data
        self.current_image_id = image_data.id
        self.initial_score = image_data.initial_score  # hier ist der initial score unser startpunkt für die Optimierung
        self.current_score = image_data.initial_score  # und dies ist dann auch der aktuelle score
        self.has_score_decreased = False
        self.mdp_active = False

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_val = int(action)
        previous_score = self.current_score
        # Wir brauchen das Bild vor der Transformation, um Dimensionsänderungen zu prüfen
        previous_image = self.current_image

        terminated = False
        truncated = False
        transformer = None
        dims_changed = False

        if self._is_stop_action(action_val):
            terminated = True
            transformer_label = "STOP"
        else:
            transformer = self.transformers[action_val]
            self.current_image, self.current_score = self._transform_and_score(self.current_image,
                                                                               transformer)
            transformer_label = transformer.label
            # Prüfen auf Dimensionsänderung (für Crop Stats)
            dims_changed = (previous_image.shape[:2] != self.current_image.shape[:2])

        # MDP (Markov Decision Process) Detection
        # Check if score went down and then up again within the episode
        if self.current_score < previous_score:
            self.has_score_decreased = True
        elif self.current_score > previous_score and self.has_score_decreased:
            self.mdp_active = True

        reward = self.reward_strategy.calculate(
            transformer_label=transformer_label,
            current_score=self.current_score,
            new_score=self.current_score,
            initial_score=self.initial_score,
            step_count=self.step_count,
            max_steps=self.max_transformations
        )

        self.step_count += 1
        
        # Truncated wenn Max Steps erreicht UND nicht explizit gestoppt wurde
        if self.step_count >= self.max_transformations and not terminated:
            truncated = True

        # History recording (analog zu ImageTransformEnv)
        score_delta = self.current_score - previous_score
        
        step_info = {
            "step": self.step_count,
            "label": transformer_label,
            "score": self.current_score,
            "reward": reward,
            "action": action,
            "transformer_type": transformer.transformer_type.name if transformer and hasattr(transformer, "transformer_type") else "UNKNOWN",
            "dims_changed": dims_changed,
            "score_delta": score_delta
        }

        self.step_history.append(step_info)

        info = {
            "score": self.current_score,
            "steps": self.step_count,
            "success": self.current_score > self.initial_score,
            "initial_score": self.initial_score,
            "action": action,
            "transformer_label": transformer_label,
            "step_history": self.step_history,
            "mdp": self.mdp_active
        }

        # Ob der reward "zwischendrin" oder nur bei "STOP" vergeben wird, entscheidet die reward strategy selber
        return self.current_image, reward , terminated, truncated, info

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
        Hier integrieren wir den Success-Bonus, um den Agenten zu motivieren, 
        erst zu verbessern und DANN zu stoppen.
        """
        reward = self.stop_bonus

        if self.current_score > self.initial_score:
            # Jackpot: Wir haben uns verbessert und loggen das Ergebnis ein.
            reward += self.success_bonus
        else:
            # Optional: Kleine Strafe für "Aufgeben ohne Versuch" (Safe Haven vermeiden)
            # Damit 0 (nichts tun) schlechter ist als "Versuchen"
            reward -= 0.1
            
        return reward
