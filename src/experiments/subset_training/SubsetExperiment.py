import logging
from pathlib import Path
from typing import List

from mlflow.entities import Experiment, Run as MlflowRun
from numpy import ndarray

from data_types.AgenticImage import ImageData
from dataset.Utils import Utils
from experiments.shared.PhotographyExperiment import PhotographyExperiment
from experiments.subset_training.TransformationActor import TransformationActor
from experiments.subset_training.ReplayBuffer import ReplayBuffer
from experiments.subset_training.DQNAgent import DQNAgent
from transformer import REVERSIBLE_TRANSFORMERS

logger = logging.getLogger(__name__)


class SubsetTraining(PhotographyExperiment):

    def __init__(self, experiment_name: str, run_name: str, dataset_root: Path, topk: int, batch_size: int):
        super(SubsetTraining, self).__init__(experiment_name)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.dataset_root = dataset_root
        self.topk = topk
        self.batch_size = batch_size
        logger.info(
            "Experiment initialized with experiment_name=%s, run_name=%s, dataset_root=%s, topk=%s, batch_size=%d",
            experiment_name, run_name, dataset_root, topk, batch_size)

        # RL specific defaults
        # Action space is a list of strings provided/overridden later; example placeholder
        self.action_space: List[str] = REVERSIBLE_TRANSFORMERS
        # desired state shape (C,H,W) for the agent
        self.state_shape = (3, 64, 64)
        self.replay_buffer = ReplayBuffer(capacity=5000)
        self.batch_size_rl = 32
        self.agent = DQNAgent(self.action_space, self.state_shape)

    def configure(self, config: dict):
        # TODO Vibecoding-Approach. Currently we configure in __init__
        # allow overriding action space and shapes from config
        if not config:
            return
        if "action_space" in config:
            self.action_space = config["action_space"]
            self.agent.action_space = self.action_space
            self.agent.n_actions = len(self.action_space)
        if "state_shape" in config:
            self.state_shape = tuple(config["state_shape"])  # e.g. [3,64,64]
        if "replay_capacity" in config:
            self.replay_buffer = ReplayBuffer(capacity=int(config["replay_capacity"]))

    # helper: convert image ndarray (H,W,C) or similar to (C,H,W) and resize/crop to state_shape
    def _preprocess_image_to_state(self, img_ndarray) -> ndarray:
        import numpy as np
        from PIL import Image

        # Convert to PIL image, resize to desired HxW, ensure 3 channels
        h_target = self.state_shape[1]
        w_target = self.state_shape[2]
        if img_ndarray is None:
            # return zeros state
            return np.zeros(self.state_shape, dtype=np.uint8)
        img = Image.fromarray(img_ndarray)
        img = img.convert("RGB")
        img = img.resize((w_target, h_target))
        arr = np.asarray(img, dtype=np.uint8)
        # arr shape is (H,W,3) -> transpose
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    # reward stub to be implemented later
    def _compute_reward(self, img_data: ImageData, action_str: str, score_after: float) -> float:
        # TODO: Sehr simple Reward-Funktion: Differenz der Scores
        #       Positiv, wenn es besser wird, Negativ wenn schlechter
        return score_after - img_data.score

    def _run_impl(self, experiment: Experiment, active_run: MlflowRun):
        dataloader = Utils.create_topk_coco_dataloader(self.dataset_root, batch_size=self.batch_size, k=int(self.topk))
        transformation_actor: TransformationActor = TransformationActor()

        self.log_param("experiment_type", "RL training with reversible transformations")
        self.log_param("dataset_root", self.dataset_root)
        self.log_param("topk", self.topk)
        self.log_param("batch_size", self.batch_size)
        self.log_param("action_space", self.action_space)

        # RL training loop over dataset images (treat each image as a single-step episode for now)
        result_csv: List[str] = ["image_id,image_relative_path,score_before,score_after,score_change,chosen_action, reward"]

        steps = 0
        for batch in dataloader:
            for img_data in batch:
                steps += 1
                logger.debug(" ImageData: id=%s, path=%s, score=%.4f", img_data.id, img_data.image_relative_path,
                             img_data.score)

                # Prepare state from original image
                cur_state = self._preprocess_image_to_state(img_data.get_image_data())

                # agent selects action (index)
                action_idx = self.agent.select_action(cur_state)
                action_str = self.agent.action_to_string(action_idx)

                # apply transformation corresponding to action (actor expects transformation name)
                # Here we assume action_str corresponds to a transformation key; user will supply mapping
                transformed_img, score_after = transformation_actor.transform_and_score(img_data.get_image_data(), action_str)

                # compute reward (stub)
                reward = self._compute_reward(img_data, action_str, score_after)

                # build next state
                next_state = self._preprocess_image_to_state(transformed_img)

                # TODO: Aktuell genau 1 Action pro Bild. Wir wollen aber mehr Actions erlauben.
                #       D.h. das Transformierte Image ist der neue state. Und für den Reward muss ich dann den score der zwischenversion nehmen.
                done = True  # single-step episode per image for this simplified scaffolding
                # push to replay buffer
                self.replay_buffer.add(cur_state, action_idx, reward, next_state, done)

                # If enough samples, perform an optimization step
                if len(self.replay_buffer) >= self.batch_size_rl:
                    batch_sample = self.replay_buffer.sample(self.batch_size_rl)
                    loss = self.agent.optimize_step(batch_sample, target_update=(steps % 100 == 0))
                    logger.debug("DQN optimize_step loss=%.6f", loss)

                logger.info("Transformation for image %s chosen_action=%s Score before=%.4f , score after=%.4f  --> reward=%.4f",
                            img_data.image_relative_path, action_str, img_data.score, score_after, reward)

                result_csv.append(
                    f"{img_data.id},{img_data.image_relative_path},{img_data.score:.4f},{score_after:.4f},{score_after - img_data.score:.4f},{action_str},{reward:.4f}")

        # Decay epsilon after all steps (check if this is desired behavior)
        self.agent.decay_epsilon()

        for line in result_csv:
            logger.info(line)
