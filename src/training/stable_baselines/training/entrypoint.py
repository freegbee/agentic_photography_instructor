import time

from training.hyperparameter_registry import HyperparameterRegistry
from training.stable_baselines.training.hyper_params import TrainingParams, DataParams, GeneralParams
from training.stable_baselines.training.trainer import StableBaselineTrainer
from transformer import POC_TWO_WAY_TRANSFORMERS
from utils.LoggingUtils import configure_logging

configure_logging()


def main():
    run_name = "run_" + time.strftime("%Y%m%d-%H%M%S")

    general_params = HyperparameterRegistry.get_store(GeneralParams)
    general_params.set({
        "success_bonus": 10.0,
        "transformer_labels": POC_TWO_WAY_TRANSFORMERS,
        "image_max_size": (384, 384)
    })

    training_params = HyperparameterRegistry.get_store(TrainingParams)
    training_params.set({
        "experiment_name": "Stable_Baselines_RL_Agent_Training PoC 0.01",
        "run_name": run_name,
        "use_local_juror": True,
        "random_seed": 42,
        "num_vector_envs": 128,
        "n_steps": 4,             #
        "mini_batch_size": 128,    # (n_steps * num_vector_env) % mini_batch_size == 0, also
        "n_epochs": 4,
        "max_transformations": 5,
        "total_training_steps": 20_000
    })

    data_params = HyperparameterRegistry.get_store(DataParams)
    data_params.set({"dataset_id": "twenty_two_actions"})

    trainer = StableBaselineTrainer()
    trainer.run_training(run_name=run_name)


if __name__ == '__main__':
    main()
