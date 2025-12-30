import time

from training.hyperparameter_registry import HyperparameterRegistry
from training.stable_baselines.models.model_factory import PpoModelVariant
from training.stable_baselines.training.hyper_params import TrainingParams, DataParams, GeneralParams
from training.stable_baselines.training.trainer import StableBaselineTrainer
from transformer import POC_MULTI_ONE_STEP_TRANSFORMERS
from utils.LoggingUtils import configure_logging
from utils.Registries import TRANSFORMER_REGISTRY

configure_logging()


def main():
    run_name = "run_" + time.strftime("%Y%m%d-%H%M%S")
    model_variant = PpoModelVariant.PPO_WITHOUT_BACKBONE
    run_name = f"Landscapes, One-of-Six, Juror Scores, {model_variant.value}"

    source_transformer_labels = POC_MULTI_ONE_STEP_TRANSFORMERS
    transformer_labels = [TRANSFORMER_REGISTRY.get(l).get_reverse_transformer_label() for l in source_transformer_labels]

    general_params = HyperparameterRegistry.get_store(GeneralParams)
    general_params.set({
        "success_bonus": 1.0,
        "learning_rate": 3e-4,
        "transformer_labels": transformer_labels,
        "image_max_size": (384, 384)
    })

    training_params = HyperparameterRegistry.get_store(TrainingParams)
    training_params.set({
        "experiment_name": "SB3_POC_EVALUATION_VISUALISATION",
        "run_name": run_name,
        "use_local_juror": True,
        "random_seed": 42,
        "ppo_model_variant": model_variant,
        "num_vector_envs": 20,
        "n_steps": 200,
        "mini_batch_size": 100,  # (n_steps * num_vector_env) % mini_batch_size == 0, also
        "n_epochs": 4,
        "max_transformations": 1,
        "total_training_steps": 400_000,
        "render_mode": "skip",  # "save",
        "render_save_dir": "./renders/",
        # evaluation parameters
        "evaluation_seed": 67,
        "evaluation_interval": 4000,  # num_vector_envs * n_steps -> Nach jedem Rollout validieren
        "evaluation_deterministic": True,
        "evaluation_render_mode": "save",
        "evaluation_render_save_dir": "./evaluation/renders/",
        "evaluation_log_path": "./evaluation/logs/",
        "evaluation_model_save_dir": "./evaluation/models/"
    })

    data_params = HyperparameterRegistry.get_store(DataParams)
    # data_params.set({"dataset_id": "lhq_landscapes_two_actions"})
    # data_params.set({"dataset_id": "lhq_landscapes_two_actions_amd-win"})
    data_params.set({"dataset_id": "lhq_landscapes_multi_one_step_actions_amd-win"})
    # data_params.set({"dataset_id": "twenty_two_actions_amd-win"})
    # data_params.set({"dataset_id": "twenty_two_actions"})

    trainer = StableBaselineTrainer()
    trainer.run_training(run_name=run_name)


if __name__ == '__main__':
    main()
