import logging
from pathlib import Path
from typing import SupportsFloat, Any, List, Tuple, Optional, Dict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete
from numpy import integer, ndarray

from data_types.AgenticImage import ImageData
from juror_client import JurorClient
from training.stable_baselines.environment.samplers import CocoDatasetSampler
from transformer.AbstractTransformer import AbstractTransformer
from utils.ImageUtils import ImageUtils

logger = logging.getLogger(__name__)


class ImageTransformEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "save"]}

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
                 render_save_dir: Path = None,
                 ):
        """
        Initialize the Image Transformation Environment.
        - Define action and observation spaces
        - manage access to transformers and images
        - setup any necessary variables or states
        - define/initialize hyperparameters
        - manage mlflow logging if needed
        """
        super(ImageTransformEnv, self).__init__()
        self.transformers = transformers
        self.coco_dataset_sampler = coco_dataset_sampler
        self.juror_client = juror_client
        self.success_bonus = success_bonus
        self.image_max_size = image_max_size
        self.max_transformations = max_transformations
        self.max_action_param_dim = max_action_param_dim

        # Observation: normalisierte Float32-Bilder
        # h, w = image_max_size
        # Observation space: 3 x h x w RGB Bild mit Werten in [0,1] fpr die einzelnen Pixel
        # self.observation_space = spaces.Box(0.0, 1.0, shape=(3, h, w), dtype=np.float32)

        # Action space: Auswahl eines Transformers + ein Parameterwert im Bereich [-1, 1]
        # transformer_index: Diskrete Auswahl eines Transformers
        # params: n (n=max_action_param_dim) mögliche Parameterwerte im Bereich [-1, 1] (kann je nach Transformer interpretiert werden)
        # Beispiel: action = {"transformer_index": 0, "params": [0.5]}
        # self.action_space = spaces.Dict({
        #     "transformer_index": spaces.Discrete(len(self.transformers)),
        #     "params": spaces.Box(-1.0, 1.0, shape=(self.max_action_param_dim,), dtype=np.float32),
        # })
        self.action_space = self._calculate_action_space()

        # State variables
        self.current_image = None
        self.initial_score = None
        self.current_score = None
        self.current_image_id = None
        self.step_count = 0
        self._rng = np.random.RandomState(seed)

        # Rendering parameters
        self.render_mode = render_mode
        self.render_save_dir = render_save_dir
        self.reset_idx = 0

        self.debug_scoring = False

    def _calculate_action_space(self) -> Discrete[integer[Any] | Any]:
        return spaces.Discrete(len(self.transformers))

    def reset(self, *, seed=None, options: Optional[Dict] = None) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment to an initial state and return the initial observation.
        - Initialize or reset the image to be transformed
        - Calculate the initial

        Implement:
        - Select random image or use a predefined one
        - preprocess image
          - resize
          - normalize
        - Return the initial observation and additional info
        """

        options = options or {}
        if options.get("reset_sampler"):
            self.coco_dataset_sampler.reset()

        self.reset_idx += 1

        if seed is not None:
            self._rng.seed(seed)
        # Informiere gymnasium über das Seed
        super().reset(seed=seed)

        # Wähle ein Bild aus dem COCO-Dataset gemäss sampler.
        # Falls der sampler erschöpft ist, wird StopIteration geworfen und training kann entsprechend reagieren
        random_index, image_data, exhausted = self.coco_dataset_sampler()
        if exhausted:
            logger.debug("Dataset sampler exhausted at reset.")
            self.coco_dataset_sampler.reset()

        self._reset_image_properties(image_data, random_index)
        self.step_count = 0
        return self.current_image, {"dataset_exhausted": exhausted}

    def _reset_image_properties(self, image_data: ImageData, random_index: int):
        img = image_data.image_data
        logger.debug(
            "Resetting environment with image id %d at index %d. Initial score=%.4f, score=%.4f" % (image_data.id,
                                                                                                    random_index,
                                                                                                    image_data.initial_score,
                                                                                                    image_data.score))
        # Die Pixelwerte des Bildes (Farben 0-255) werden in den Bereich [0,1] normalisiert
        # Beim Scoren wird dann wieder zurückgerechnet
        self.current_image = img
        self.current_image_id = image_data.id

        self.initial_score = image_data.initial_score
        self.current_score = image_data.score
        if self.debug_scoring:
            life_score = self.juror_client.score_ndarray_bgr(img)
            if life_score.score != self.current_score:
                logger.warning("Initial score mismatch for image id %d: stored=%.4f, live=%.4f" % (image_data.id,
                                                                                                   self.current_score,
                                                                                                   life_score.score))
                proof_file_name = str(image_data.image_path).replace(".png", "_with_mismatch.png")
                proof_file_name = str(proof_file_name).replace(".jpg", "_with_mismatch.jpg")
                ImageUtils.save_image(image_data.image_data, proof_file_name)
                raise (ValueError("Initial score mismatch. proof at %s" % proof_file_name))

    def reset_sampler(self):
        self.coco_dataset_sampler.reset()


    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one time step within the environment.
        - Apply the action to the environment (i.e. the image)
        - Calculate the reward based on the action taken
        - Determine if the episode has ended
        - Return the new observation, reward, terminated flag, truncated flag and additional info
          - observation: the transformed image after applying the action
          - reward: a float value representing the reward obtained from the action
          - terminated: a boolean flag indicating if the episode has ended
          - truncated: a boolean flag indicating if the episode was truncated
          - info: a dictionary containing any additional information
        """

        temp_previous_score = self.current_score

        if isinstance(action, dict):
            # Falls die Action parametriert ist
            transformer_index = action["transformer_index"]
            params = np.asarray(action["params"], dtype=np.float32)
        else:
            # Parameterfreie Actions
            transformer_index = int(action)
            params = np.array([], dtype=np.float32)
        transformer = self.transformers[transformer_index]

        # Wende den Transformer auf das aktuelle Bild an
        transformed_img = transformer.transform(self.current_image)

        # Berechne die neue Punktzahl mit dem Juror-Client
        scoring_response = self.juror_client.score_ndarray_bgr(transformed_img)
        new_score = scoring_response.score

        # Berechne die Belohnung als Differenz der Punktzahlen
        reward = new_score - self.current_score

        # optional: Penalty für große params oder jeden Schritt
        # param_penalty = 0.01 * float(np.linalg.norm(params))
        # step_penalty = -0.001
        # reward = reward - param_penalty + step_penalty

        # Erfolg prüfen und Bonus vergeben
        success = (new_score >= self.initial_score) if (self.initial_score is not None) else False
        if not success and abs(new_score - self.initial_score) < 0.25:
            logger.warning(
                f"Suspiciously small score change for image id {self.current_image_id}: initial={self.initial_score}, new={new_score}")
        if success:
            reward += self.success_bonus

        # Aktualisiere den Zustand (bild ist wieder normalisiert im Bereich [0,1])
        # self.current_image = self._preprocess(transformed_img)
        self.current_image = transformed_img
        self.current_score = new_score
        self.step_count += 1

        # TODO: Prüfen, ob es eine "truncated" Bedingung gibt und diese implementieren
        # truncated = not success and (self.step_count >= self.max_transformations)
        # Nie truncated, da das korrekterweise fertig ist und das Netz nicht meinen soll, dass es eigentlich noch hätte weiter gehen sollen
        truncated = False  # Success oder Anzahl Steps gemacht, d.h. 2
        done = True  # Success oder truncated

        info = {
            "score": new_score,
            # "param_penalty": param_penalty,
            "steps": self.step_count,
            "success": bool(success),
            "initial_score": self.initial_score
        }

        logger.debug(
            "Step: %d: Applied action %d (transformer %s) to image %d. Score %.4f -> %.4f" % (self.step_count, action,
                                                                                              transformer.label,
                                                                                              self.current_image_id,
                                                                                              temp_previous_score,
                                                                                              self.current_score))

        return self.current_image, reward, done, truncated, info

    def _transform_and_score(self, image_data: ndarray, transformer: AbstractTransformer) -> Tuple[ndarray, float]:
        transformed_img = transformer.transform(self.current_image)
        scoring_response = self.juror_client.score_ndarray_bgr(transformed_img)
        return transformed_img, scoring_response.score


    def close(self):
        """
        Clean up resources, e.g. file handles, connections, plots, etc.

        Implement this method to release any resources held by the environment after training and testing are complete.
        Terminate mlflow experiment
        """
        try:
            plt.close("all")
        except Exception:
            pass
