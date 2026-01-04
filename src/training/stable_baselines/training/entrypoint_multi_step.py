"""
Example entrypoint for training with multi-step transformation wrapper.

This demonstrates how to configure the training to apply multiple transformations
per episode, with reward only given after all transformations are applied.
"""
import time

from training.hyperparameter_registry import HyperparameterRegistry
from training.stable_baselines.models.model_factory import PpoModelVariant
from training.stable_baselines.hyperparameter.training_hyperparams import TrainingParams
from training.stable_baselines.hyperparameter.general_hyperparams import GeneralParams
from training.stable_baselines.hyperparameter.data_hyperparams import DataParams
from training.stable_baselines.training.trainer import StableBaselineTrainer
from transformer import POC_MULTI_ONE_STEP_TRANSFORMERS, POC_MULTI_TWO_STEP_TRANSFORMERS
from utils.LoggingUtils import configure_logging
from utils.Registries import TRANSFORMER_REGISTRY

configure_logging()


def main():
    """
    Train an RL agent to apply multiple transformations before receiving reward.

    Key differences from single-step training:
    - use_multi_step_wrapper=True enables the wrapper
    - steps_per_episode=2 means agent must take 2 actions per episode
    - Reward is only given after both transformations are applied
    - Agent learns to plan multi-step transformation sequences
    """
    model_variant = PpoModelVariant.PPO_WITHOUT_BACKBONE
    run_name = f"{time.strftime('%Y%m%d-%H%M%S')} - Landscapes, Two-of-Six, Juror Scores, {model_variant.value}"

    # Use reversing transformers (to undo degradations)
    source_transformer_labels = POC_MULTI_TWO_STEP_TRANSFORMERS
    transformer_labels = [TRANSFORMER_REGISTRY.get(l).get_reverse_transformer_label() for l in
                          source_transformer_labels]

    general_params = HyperparameterRegistry.get_store(GeneralParams)
    general_params.set({
        "success_bonus": 1.0,
        "learning_rate": 1e-4,
        "transformer_labels": transformer_labels,
        "image_max_size": (384, 384)
    })

    training_params = HyperparameterRegistry.get_store(TrainingParams)
    training_params.set({
        "experiment_name": "SB3_MULTI_STEP_TRAINING",
        "run_name": run_name,
        "use_local_juror": True,
        "random_seed": 42,
        "ppo_model_variant": model_variant,
        "num_vector_envs": 20,
        "n_steps": 200,
        "mini_batch_size": 100, # (n_steps * num_vector_env) % mini_batch_size == 0, also
        "n_epochs": 4,
        "max_transformations": 2,  # Maximum transformations allowed
        "total_training_steps": 600_000,
        "render_mode": "skip",
        "render_save_dir": "./renders/",

        # === MULTI-STEP WRAPPER CONFIGURATION ===
        "use_multi_step_wrapper": True,  # Enable multi-step wrapper
        "steps_per_episode": 2,  # Agent must take 2 actions per episode
        "multi_step_intermediate_reward": False,  # No reward for intermediate steps (default)
        "multi_step_reward_shaping": False,  # No shaped rewards (default)
        # =========================================

        # Evaluation parameters
        "evaluation_seed": 67,
        "evaluation_interval": 4000, # num_vector_envs * n_steps -> Nach jedem Rollout validieren
        "evaluation_deterministic": True,
        "evaluation_visual_history": True,
        "evaluation_visual_history_max_images": 15,
        "evaluation_visual_history_max_size": 150,
        "evaluation_render_mode": "skip",
        "evaluation_render_save_dir": "./evaluation/renders/",
        "evaluation_log_path": "./evaluation/logs/",
        "evaluation_model_save_dir": "./evaluation/models/"
    })

    data_params = HyperparameterRegistry.get_store(DataParams)
    # Use a dataset where images have 2 degradations applied
    data_params.set({"dataset_id": "lhq_landscapes_multi_two_step_actions_amd-mac"})

    trainer = StableBaselineTrainer()
    trainer.run_training(run_name=run_name)


def main_with_reward_shaping():
    """
    Alternative configuration with reward shaping for intermediate steps.

    This provides small shaped rewards at intermediate steps to help guide learning.
    Use this if the agent struggles to learn with purely delayed rewards.
    """
    model_variant = PpoModelVariant.PPO_WITHOUT_BACKBONE
    run_name = f"{time.strftime('%Y%m%d-%H%M%S')} - Multi-Step with Shaping, {model_variant.value}"

    source_transformer_labels = POC_MULTI_ONE_STEP_TRANSFORMERS
    transformer_labels = [TRANSFORMER_REGISTRY.get(l).get_reverse_transformer_label() for l in
                          source_transformer_labels]

    general_params = HyperparameterRegistry.get_store(GeneralParams)
    general_params.set({
        "success_bonus": 1.0,
        "learning_rate": 3e-4,
        "transformer_labels": transformer_labels,
        "image_max_size": (384, 384)
    })

    training_params = HyperparameterRegistry.get_store(TrainingParams)
    training_params.set({
        "experiment_name": "SB3_MULTI_STEP_WITH_SHAPING",
        "run_name": run_name,
        "use_local_juror": True,
        "random_seed": 42,
        "ppo_model_variant": model_variant,
        "num_vector_envs": 20,
        "n_steps": 200,
        "mini_batch_size": 100,
        "n_epochs": 4,
        "max_transformations": 2,
        "total_training_steps": 400_000,
        "render_mode": "skip",
        "render_save_dir": "./renders/",

        # === MULTI-STEP WITH REWARD SHAPING ===
        "use_multi_step_wrapper": True,
        "steps_per_episode": 2,
        "multi_step_intermediate_reward": False,
        "multi_step_reward_shaping": True,  # Enable shaped rewards for guidance
        # ========================================

        "evaluation_seed": 67,
        "evaluation_interval": 4000,
        "evaluation_deterministic": True,
        "evaluation_visual_history": True,
        "evaluation_visual_history_max_images": 15,
        "evaluation_visual_history_max_size": 150,
        "evaluation_render_mode": "skip",
        "evaluation_render_save_dir": "./evaluation/renders/",
        "evaluation_log_path": "./evaluation/logs/",
        "evaluation_model_save_dir": "./evaluation/models/"
    })

    data_params = HyperparameterRegistry.get_store(DataParams)
    data_params.set({"dataset_id": "lhq_landscapes_multi_two_step_actions_amd-mac"})

    trainer = StableBaselineTrainer()
    trainer.run_training(run_name=run_name)


if __name__ == '__main__':
    # Choose which configuration to run:
    main()  # Pure delayed reward (recommended for 2 steps)
    # main_with_reward_shaping()  # With shaped rewards (if learning is difficult)
