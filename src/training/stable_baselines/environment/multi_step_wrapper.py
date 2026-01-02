"""
Multi-Step Transformation Wrapper for RL Environment.

This wrapper allows an agent to take multiple transformation actions before receiving a reward.
Only after N steps does the agent receive a reward based on the cumulative effect of all transformations.
"""
import logging
from typing import SupportsFloat, Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

logger = logging.getLogger(__name__)


class MultiStepTransformWrapper(gym.Wrapper):
    """
    Wraps an ImageTransformEnv to support multi-step episodes.

    Instead of receiving reward after each transformation, the agent must take
    N transformation actions before receiving a reward based on the final result.

    Key features:
    - Buffers N actions before calculating reward
    - Returns zero reward for intermediate steps
    - Only evaluates final image after N transformations
    - Tracks transformation sequence for analysis

    Args:
        env: The base ImageTransformEnv to wrap
        steps_per_episode: Number of transformations to apply before episode ends
        intermediate_reward: Whether to provide small intermediate rewards (default: False)
        reward_shaping: Whether to provide shaped rewards based on progress (default: False)
    """

    def __init__(
        self,
        env: gym.Env,
        steps_per_episode: int = 2,
        intermediate_reward: bool = False,
        reward_shaping: bool = False
    ):
        super().__init__(env)
        self.steps_per_episode = steps_per_episode
        self.intermediate_reward = intermediate_reward
        self.reward_shaping = reward_shaping

        # State tracking
        self.current_step = 0
        self.transformation_sequence = []
        self.intermediate_scores = []
        self.initial_score = None

    def reset(self, **kwargs) -> Tuple[ObsType, dict[str, Any]]:
        """Reset the environment and tracking variables."""
        observation, info = self.env.reset(**kwargs)

        # Reset tracking
        self.current_step = 0
        self.transformation_sequence = []
        self.intermediate_scores = []
        self.initial_score = self.env.initial_score

        logger.debug(f"Multi-step wrapper reset. Episode will run for {self.steps_per_episode} steps.")

        return observation, info

    def step(self, action) -> Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one transformation step.

        For steps 1 to N-1:
        - Apply transformation
        - Return zero (or small shaped) reward
        - Episode continues (done=False)

        For step N:
        - Apply final transformation
        - Calculate reward based on improvement from initial score
        - Episode ends (done=True)
        """
        self.current_step += 1
        is_final_step = (self.current_step >= self.steps_per_episode)

        # Get transformer label for tracking
        if isinstance(action, dict):
            transformer_index = action["transformer_index"]
        else:
            transformer_index = int(action)

        transformer_label = self.env.transformers[transformer_index].label
        self.transformation_sequence.append(transformer_label)

        # Apply transformation (this updates env.current_image and env.current_score)
        observation, base_reward, base_done, base_truncated, info = self.env.step(action)

        # Store intermediate score
        current_score = self.env.current_score
        self.intermediate_scores.append(current_score)

        if not is_final_step:
            # Intermediate step - don't give final reward yet
            if self.intermediate_reward:
                # Small reward for improvement at each step
                reward = base_reward * 0.1  # Scale down intermediate rewards
            elif self.reward_shaping:
                # Shaped reward: small positive if moving toward goal
                score_improvement = current_score - self.initial_score
                reward = np.clip(score_improvement * 0.05, -0.1, 0.1)  # Small shaped reward
            else:
                # No reward for intermediate steps
                reward = 0.0

            # Episode continues
            done = False
            truncated = False

            logger.debug(
                f"Multi-step: Step {self.current_step}/{self.steps_per_episode} - "
                f"Applied {transformer_label}, Score: {current_score:.4f}, "
                f"Intermediate reward: {reward:.4f}"
            )

        else:
            # Final step - calculate reward based on total improvement
            final_score = current_score
            total_improvement = final_score - self.initial_score

            # Reward is based on total improvement from initial score
            reward = total_improvement

            # Check if we successfully improved the image
            success = final_score >= self.initial_score

            # Add success bonus (from base environment)
            if success:
                reward += self.env.success_bonus

            # Episode ends
            done = True
            truncated = False

            logger.info(
                f"Multi-step: Episode complete after {self.current_step} steps. "
                f"Initial: {self.initial_score:.4f}, Final: {final_score:.4f}, "
                f"Improvement: {total_improvement:.4f}, Success: {success}, "
                f"Final reward: {reward:.4f}"
            )
            logger.info(f"Transformation sequence: {' -> '.join(self.transformation_sequence)}")

            # Update info with multi-step statistics
            info["multi_step"] = {
                "steps_taken": self.current_step,
                "transformation_sequence": self.transformation_sequence.copy(),
                "intermediate_scores": self.intermediate_scores.copy(),
                "initial_score": self.initial_score,
                "final_score": final_score,
                "total_improvement": total_improvement,
                "success": success
            }

        return observation, reward, done, truncated, info

    def close(self):
        """Clean up resources."""
        return self.env.close()
