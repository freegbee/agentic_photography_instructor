import time

from utils.LoggingUtils import configure_logging

# DQN profitiert in der Regel nicht von Multiprocessing (Sample Efficiency vs Wall Clock Time).
# Zudem vereinfacht Single-Processing das Debugging.
NUM_VECTOR_ENVS = 1

from training.stable_baselines.environment.welldefined_environments import WellDefinedEnvironment
from training.hyperparameter_registry import HyperparameterRegistry
from training.stable_baselines.hyperparameter.runtime_hyperparams import RuntimeParams
from training.stable_baselines.hyperparameter.runtime_params_builder import RuntimeParamsBuilder
from training.stable_baselines.hyperparameter.task_hyperparams import TaskParams
from training.stable_baselines.hyperparameter.task_params_builder import TaskParamsBuilder
from training.stable_baselines.hyperparameter.data_hyperparams import DataParams
from training.stable_baselines.training.trainer import StableBaselineTrainer
from transformer import SENSIBLE_TRANSFORMERS

# DQN Imports
from training.stable_baselines.models.dqn_model_variants import DqnModelVariant
from training.stable_baselines.hyperparameter.dqn_model_hyperparams import DqnModelParams
from training.stable_baselines.hyperparameter.dqn_params_builder import DqnParamsBuilder
from training.stable_baselines.models.dqn_model_factory import DqnModelFactory

configure_logging()


def main():
    # ========================================================================================
    # 1. EXPERIMENT DEFINITION (THE "WHAT")
    # Hier definieren wir die fachlichen Parameter des Experiments.
    # ========================================================================================
    experiment_name = "SB3_POC_DQN_IMAGE_OPTIMIZATION"
    run_description = "DQN, Landscapes, Sensible"

    dataset_id = "twenty_original_split_amd-win"
    image_size = (384, 384)
    # Für debug wird 1/10 verwendet. Definiert die Grösse des replay buffers
    # Siehe training.stable_baselines.hyperparameter.dqn_params_builder.DqnParamsBuilder.calculate_buffer_size
    target_memory_mb=8_000
    core_env = WellDefinedEnvironment.IMAGE_OPTIMIZATION
    transformer_labels = SENSIBLE_TRANSFORMERS
    max_transformations = 10

    # DQN Modell Konfiguration
    model_variant = DqnModelVariant.DQN_WITHOUT_BACKBONE
    learning_rate = 1e-4

    # ========================================================================================
    # 2. EXECUTION MODE (THE "HOW")
    # Hier steuern wir technische Parameter für Debugging vs. echtes Training.
    # ========================================================================================
    IS_DEBUG_RUN = True

    if IS_DEBUG_RUN:
        print("\n!!! RUNNING IN DEBUG MODE !!!\n")
        buffer_size = DqnParamsBuilder.calculate_buffer_size(image_size, target_memory_mb=int(target_memory_mb/10))
        batch_size = 32
        learning_starts = 100
        total_training_steps = 2000
        evaluation_interval = 500
        run_name_prefix = f"DEBUG - {run_description}"
    else:
        buffer_size = DqnParamsBuilder.calculate_buffer_size(image_size, target_memory_mb=target_memory_mb)
        batch_size = 32
        learning_starts = 1000
        total_training_steps = 300_000
        evaluation_interval = 5000
        run_name_prefix = run_description

    # ========================================================================================
    # 3. CALCULATION & ASSEMBLY
    # Berechnung abgeleiteter Werte und Befüllen der Builder.
    # ========================================================================================
    print(f"Calculated buffer_size: {buffer_size} transitions based on image size {image_size}")
    run_name = f"{time.strftime('%Y%m%d-%H%M%S')} - {run_name_prefix}, {model_variant.value}"

    # 3.1 DQN Model Parameters
    dqn_params = HyperparameterRegistry.get_store(DqnModelParams)
    dqn_config = (DqnParamsBuilder(variant=model_variant,
                                   buffer_size=buffer_size,
                                   batch_size=batch_size,
                                   learning_rate=learning_rate)
                  # train_freq=1: Update nach jedem Schritt (wichtig bei kurzen Episoden von nur 10 Steps)
                  .with_training_schedule(learning_starts=learning_starts,
                                          train_freq=1,
                                          gradient_steps=1,
                                          target_update_interval=1000)
                  .with_exploration(fraction=0.1, initial_eps=1.0, final_eps=0.05)
                  .build())
    dqn_params.set(dqn_config)

    # 3.2 Task Parameters
    task_params = HyperparameterRegistry.get_store(TaskParams)
    task_config = (TaskParamsBuilder(core_env=core_env,
                                     transformer_labels=transformer_labels,
                                     max_transformations=max_transformations)
                   .with_rewards(success_bonus=1.0)
                   .build())
    task_params.set(task_config)

    # 3.3 Runtime Parameters
    runtime_params = HyperparameterRegistry.get_store(RuntimeParams)
    runtime_config = (RuntimeParamsBuilder(experiment_name=experiment_name,
                                           run_name=run_name,
                                           total_training_steps=total_training_steps,
                                           num_vector_envs=NUM_VECTOR_ENVS)
                      .with_resource_settings(use_worker_pool=False,
                                              num_juror_workers=0,
                                              vec_env_cls="DummyVecEnv")
                      .with_evaluation(interval=evaluation_interval, visual_history=True)
                      .build())
    runtime_params.set(runtime_config)

    # 3.4 Data Parameters
    data_params = HyperparameterRegistry.get_store(DataParams)
    data_params.set({"dataset_id": dataset_id, "image_max_size": image_size})

    # ========================================================================================
    # 4. START
    # Berechnung abgeleiteter Werte und Befüllen der Builder.
    # ========================================================================================
    trainer = StableBaselineTrainer(model_factory=DqnModelFactory(),
                                    model_params_class=DqnModelParams)
    trainer.run_training(run_name=run_name)

if __name__ == '__main__':
    main()