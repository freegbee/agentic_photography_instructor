import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from mlflow.entities import Experiment, Run as MlflowRun
from torch.utils.data import DataLoader

from data_types.AgenticImage import ImageData
from experiments.rl_training.RLDataset import RLDataset
from experiments.shared.PhotographyExperiment import PhotographyExperiment
from experiments.subset_training.DQNAgent import DQNAgent
from experiments.subset_training.ReplayBuffer import ReplayBuffer
from experiments.subset_training.TransformationActor import TransformationActor
from transformer import REVERSIBLE_TRANSFORMERS

logger = logging.getLogger(__name__)


class RLTrainingExperiment(PhotographyExperiment):
    """
    Reinforcement Learning training experiment for image optimization.

    Uses DQN to learn which transformations improve image aesthetic scores.
    Supports multi-step episodes where the agent can apply sequences of transformations.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str],
        dataset_root: Path,
        # RL hyperparameters
        max_steps_per_episode: int = 5,
        batch_size: int = 32,
        replay_capacity: int = 10000,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_frequency: int = 100,
        # Training parameters
        num_epochs: int = 10,
        dataloader_batch_size: int = 8,
        validation_frequency: int = 1,
        # Reward shaping
        step_penalty: float = 0.01,
        terminal_bonus: float = 0.1,
        # Action space
        action_space: List[str] = None,
        add_stop_action: bool = True,
        # State configuration
        state_shape: Tuple[int, int, int] = (3, 64, 64),
    ):
        super().__init__(experiment_name)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.dataset_root = dataset_root

        # RL configuration
        self.max_steps_per_episode = max_steps_per_episode
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency

        # Training configuration
        self.num_epochs = num_epochs
        self.dataloader_batch_size = dataloader_batch_size
        self.validation_frequency = validation_frequency

        # Reward shaping
        self.step_penalty = step_penalty
        self.terminal_bonus = terminal_bonus

        # Action space setup
        if action_space is None:
            self.action_space = REVERSIBLE_TRANSFORMERS.copy()
        else:
            self.action_space = action_space

        self.add_stop_action = add_stop_action
        if add_stop_action:
            self.action_space.append("STOP")

        # State configuration
        self.state_shape = state_shape

        # Initialize components
        self.replay_buffer = ReplayBuffer(capacity=self.replay_capacity)
        self.agent = DQNAgent(
            self.action_space,
            self.state_shape,
            lr=self.learning_rate,
            gamma=self.gamma
        )
        self.agent.epsilon = self.epsilon_start
        self.agent.epsilon_min = self.epsilon_end
        self.agent.epsilon_decay = self.epsilon_decay

        self.transformation_actor = TransformationActor()

        logger.info(
            "Initialized RLTrainingExperiment: max_steps=%d, replay_capacity=%d, action_space_size=%d",
            self.max_steps_per_episode, self.replay_capacity, len(self.action_space)
        )

    def configure(self, config: dict):
        """Configure experiment from dictionary (for flexibility)."""
        if not config:
            return

        if "max_steps_per_episode" in config:
            self.max_steps_per_episode = config["max_steps_per_episode"]
        if "batch_size" in config:
            self.batch_size = config["batch_size"]
        if "num_epochs" in config:
            self.num_epochs = config["num_epochs"]

    def _get_run_name(self) -> Optional[str]:
        return self.run_name

    def _run_impl(self, experiment: Experiment, active_run: MlflowRun):
        """Main training loop."""
        # Log all hyperparameters
        self._log_hyperparameters()

        # Load datasets
        train_dataset, val_dataset = self._load_datasets()

        # Training loop
        global_step = 0
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            self.log_metric("epoch", epoch, step=global_step)

            # Training phase
            train_metrics = self._train_epoch(train_dataset, epoch, global_step)
            global_step += train_metrics["steps"]

            # Log training metrics
            for key, value in train_metrics.items():
                self.log_metric(f"train_{key}", value, step=global_step)

            # Validation phase
            if (epoch + 1) % self.validation_frequency == 0:
                val_metrics = self._validate(val_dataset, global_step)
                for key, value in val_metrics.items():
                    self.log_metric(f"val_{key}", value, step=global_step)

            # Save checkpoint
            self._save_checkpoint(epoch, global_step)

            # Decay epsilon
            self.agent.decay_epsilon()
            self.log_metric("epsilon", self.agent.epsilon, step=global_step)

        logger.info("Training completed")

    def _log_hyperparameters(self):
        """Log all hyperparameters to MLflow."""
        self.log_param("experiment_type", "RL image optimization")
        self.log_param("dataset_root", str(self.dataset_root))
        self.log_param("max_steps_per_episode", self.max_steps_per_episode)
        self.log_param("batch_size", self.batch_size)
        self.log_param("replay_capacity", self.replay_capacity)
        self.log_param("learning_rate", self.learning_rate)
        self.log_param("gamma", self.gamma)
        self.log_param("epsilon_start", self.epsilon_start)
        self.log_param("epsilon_end", self.epsilon_end)
        self.log_param("epsilon_decay", self.epsilon_decay)
        self.log_param("target_update_frequency", self.target_update_frequency)
        self.log_param("num_epochs", self.num_epochs)
        self.log_param("step_penalty", self.step_penalty)
        self.log_param("terminal_bonus", self.terminal_bonus)
        self.log_param("add_stop_action", self.add_stop_action)
        self.log_param("action_space_size", len(self.action_space))
        self.log_param("state_shape", str(self.state_shape))

    def _load_datasets(self) -> Tuple[RLDataset, RLDataset]:
        """Load train and validation datasets."""
        train_images_root = self.dataset_root / "train" / "images"
        train_annotations = self.dataset_root / "train" / "annotations.json"
        val_images_root = self.dataset_root / "val" / "images"
        val_annotations = self.dataset_root / "val" / "annotations.json"

        train_dataset = RLDataset(train_images_root, train_annotations)
        val_dataset = RLDataset(val_images_root, val_annotations)

        logger.info(f"Loaded train dataset: {len(train_dataset)} samples")
        logger.info(f"Loaded val dataset: {len(val_dataset)} samples")

        self.log_param("train_size", len(train_dataset))
        self.log_param("val_size", len(val_dataset))

        return train_dataset, val_dataset

    def _train_epoch(self, dataset: RLDataset, epoch: int, global_step: int) -> dict:
        """Train for one epoch."""
        dataloader = DataLoader(dataset, batch_size=self.dataloader_batch_size, shuffle=True)

        total_reward = 0.0
        total_steps = 0
        total_episodes = 0
        total_loss = 0.0
        num_optimization_steps = 0
        successful_episodes = 0  # Episodes that improved the score

        for batch_idx, batch in enumerate(dataloader):
            for degraded_image_data, original_score, transformation in batch:
                # Run one episode
                episode_reward, episode_steps, final_score, episode_loss = self._run_episode(
                    degraded_image_data, original_score, global_step + total_steps
                )

                total_reward += episode_reward
                total_steps += episode_steps
                total_episodes += 1

                if final_score > degraded_image_data.score:
                    successful_episodes += 1

                # Optimization steps
                if len(self.replay_buffer) >= self.batch_size:
                    batch_sample = self.replay_buffer.sample(self.batch_size)
                    target_update = (total_steps % self.target_update_frequency == 0)
                    loss = self.agent.optimize_step(batch_sample, target_update=target_update)
                    total_loss += loss
                    num_optimization_steps += 1

                if (total_episodes % 10 == 0):
                    logger.info(
                        f"Epoch {epoch}, Episode {total_episodes}: reward={episode_reward:.4f}, "
                        f"steps={episode_steps}, score_improvement={final_score - degraded_image_data.score:.4f}"
                    )

        avg_reward = total_reward / total_episodes if total_episodes > 0 else 0.0
        avg_steps = total_steps / total_episodes if total_episodes > 0 else 0.0
        avg_loss = total_loss / num_optimization_steps if num_optimization_steps > 0 else 0.0
        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0

        return {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "avg_loss": avg_loss,
            "success_rate": success_rate,
            "total_episodes": total_episodes,
            "steps": total_steps,
        }

    def _run_episode(self, image_data: ImageData, target_score: float, global_step: int) -> Tuple[float, int, float, float]:
        """
        Run one episode (multi-step transformation sequence).

        Returns:
            (total_reward, num_steps, final_score, avg_loss)
        """
        current_image = image_data.get_image_data("BGR").copy()
        current_score = image_data.score
        initial_score = current_score

        total_reward = 0.0
        episode_steps = 0

        for step in range(self.max_steps_per_episode):
            # Preprocess state
            state = self._preprocess_image_to_state(current_image)

            # Select action
            action_idx = self.agent.select_action(state)
            action_str = self.agent.action_to_string(action_idx)

            # Check for stop action
            if self.add_stop_action and action_str == "STOP":
                # Agent chose to stop
                reward = self._compute_reward(current_score, current_score, target_score, terminal=True)
                total_reward += reward
                break

            # Apply transformation
            try:
                transformed_image, new_score = self.transformation_actor.transform_and_score(
                    current_image, action_str
                )
            except Exception as e:
                logger.warning(f"Transformation {action_str} failed: {e}")
                # Penalize failed transformations
                reward = -1.0
                next_state = state
                done = True
                self.replay_buffer.add(state, action_idx, reward, next_state, done)
                total_reward += reward
                break

            # Compute reward
            is_terminal = (step == self.max_steps_per_episode - 1)
            reward = self._compute_reward(current_score, new_score, target_score, terminal=is_terminal)
            total_reward += reward

            # Next state
            next_state = self._preprocess_image_to_state(transformed_image)

            # Store transition
            done = is_terminal
            self.replay_buffer.add(state, action_idx, reward, next_state, done)

            # Update for next iteration
            current_image = transformed_image
            current_score = new_score
            episode_steps += 1

            if is_terminal:
                break

        final_score = current_score
        return total_reward, episode_steps, final_score, 0.0

    def _validate(self, dataset: RLDataset, global_step: int) -> dict:
        """Validate the agent without exploration."""
        # Save current epsilon and set to 0 for greedy evaluation
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        dataloader = DataLoader(dataset, batch_size=self.dataloader_batch_size, shuffle=False)

        total_reward = 0.0
        total_improvement = 0.0
        total_episodes = 0
        successful_episodes = 0

        for batch in dataloader:
            for degraded_image_data, original_score, transformation in batch:
                episode_reward, episode_steps, final_score, _ = self._run_episode(
                    degraded_image_data, original_score, global_step
                )

                improvement = final_score - degraded_image_data.score
                total_reward += episode_reward
                total_improvement += improvement
                total_episodes += 1

                if improvement > 0:
                    successful_episodes += 1

        # Restore epsilon
        self.agent.epsilon = original_epsilon

        avg_reward = total_reward / total_episodes if total_episodes > 0 else 0.0
        avg_improvement = total_improvement / total_episodes if total_episodes > 0 else 0.0
        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0

        logger.info(
            f"Validation: avg_reward={avg_reward:.4f}, avg_improvement={avg_improvement:.4f}, "
            f"success_rate={success_rate:.4f}"
        )

        return {
            "avg_reward": avg_reward,
            "avg_improvement": avg_improvement,
            "success_rate": success_rate,
        }

    def _compute_reward(self, prev_score: float, new_score: float, target_score: float, terminal: bool = False) -> float:
        """
        Compute shaped reward for a transition.

        Components:
        - Score delta (main signal)
        - Step penalty (encourages efficiency)
        - Terminal bonus (if we reached or exceeded target)
        """
        score_delta = new_score - prev_score
        reward = score_delta

        # Step penalty
        reward -= self.step_penalty

        # Terminal bonus if we reached target
        if terminal and new_score >= target_score:
            reward += self.terminal_bonus

        return reward

    def _preprocess_image_to_state(self, img_ndarray: np.ndarray) -> np.ndarray:
        """Convert image to state representation (C, H, W) with proper shape."""
        from PIL import Image

        h_target, w_target = self.state_shape[1], self.state_shape[2]

        if img_ndarray is None:
            return np.zeros(self.state_shape, dtype=np.uint8)

        # OpenCV images are BGR, convert to RGB for PIL
        import cv2
        img_rgb = cv2.cvtColor(img_ndarray, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img_rgb)
        img = img.resize((w_target, h_target))
        arr = np.asarray(img, dtype=np.uint8)

        # Transpose from (H, W, C) to (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def _save_checkpoint(self, epoch: int, global_step: int):
        """Save model checkpoint."""
        import torch

        checkpoint_dir = Path("/tmp") / "rl_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'policy_net_state_dict': self.agent.policy_net.state_dict(),
            'target_net_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon,
        }, checkpoint_path)

        # Log to MLflow
        self.log_artifact(str(checkpoint_path))
        logger.info(f"Saved checkpoint to {checkpoint_path}")
