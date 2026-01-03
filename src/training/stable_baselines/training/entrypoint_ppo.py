import os
import time

# Konfiguration basierend auf Benchmark-Ergebnissen (8 Envs war am schnellsten auf 16 Cores)
OPTIMIZE_FOR_MULTIPROCESSING = True
NUM_VECTOR_ENVS = 8

if OPTIMIZE_FOR_MULTIPROCESSING:
    # WICHTIG: Threading-Limitierung VOR allen anderen Imports setzen.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

from training.stable_baselines.environment.welldefined_environments import WellDefinedEnvironment
from training.hyperparameter_registry import HyperparameterRegistry
from training.stable_baselines.models.model_variants import PpoModelVariant
from training.stable_baselines.hyperparameter.runtime_hyperparams import RuntimeParams
from training.stable_baselines.hyperparameter.runtime_params_builder import RuntimeParamsBuilder
from training.stable_baselines.hyperparameter.task_hyperparams import TaskParams
from training.stable_baselines.hyperparameter.task_params_builder import TaskParamsBuilder
from training.stable_baselines.hyperparameter.ppo_model_hyperparams import PpoModelParams
from training.stable_baselines.hyperparameter.ppo_model_hyperparams_builder import PpoModelParamsBuilder
from training.stable_baselines.hyperparameter.data_hyperparams import DataParams
from training.stable_baselines.training.trainer import StableBaselineTrainer
from transformer import SENSIBLE_TRANSFORMERS
from utils.LoggingUtils import configure_logging


def calculate_n_steps(num_envs, target_total=4000, batch_size=100):
    """
    Berechnet n_steps so, dass die totale Anzahl Steps (n_steps * num_envs)
    nahe am target_total liegt UND durch batch_size teilbar ist (wichtig für PPO).
    """
    ideal_n = target_total // num_envs

    # Suche nach dem nächsten passenden Wert (aufwärts und abwärts)
    for i in range(1000):
        # Check up
        n = ideal_n + i
        if (n * num_envs) % batch_size == 0:
            return n
        # Check down
        n = ideal_n - i
        if n > 0 and (n * num_envs) % batch_size == 0:
            return n
    return ideal_n


def main():
    configure_logging()
    model_variant = PpoModelVariant.PPO_WITHOUT_BACKBONE

    opt_str = "MultiProc" if OPTIMIZE_FOR_MULTIPROCESSING else "SingleProc"

    # Setup basierend auf Modus
    # Beispiel für das neue Setup
    run_name_prefix = "Landscapes, Multi-Optimization, Sensible"
    core_env = WellDefinedEnvironment.IMAGE_OPTIMIZATION
    transformer_labels = SENSIBLE_TRANSFORMERS
    dataset_id = "twenty_original_split_amd-win"
    # dataset_id = "flickr2k_big_original_split_amd-win"
    # dataset_id = "lhq_landscapes_original_split_amd-win"
    max_transformations = 10
    # core_env_cls_name = "ImageTransformEnvVariant"

    run_name = f"{time.strftime('%Y%m%d-%H%M%S')} - {run_name_prefix}, {model_variant.value} ({opt_str})"

    # Dynamische Berechnung von n_steps für konstante Batch-Größe (~4000)
    target_rollout_size = 4000
    batch_size = 100
    n_steps = calculate_n_steps(NUM_VECTOR_ENVS, target_rollout_size, batch_size=batch_size)

    print(f"Configuration: Envs={NUM_VECTOR_ENVS}, n_steps={n_steps} (Total Rollout={n_steps * NUM_VECTOR_ENVS})")

    ppo_params = HyperparameterRegistry.get_store(PpoModelParams)
    
    # Verwendung des Builders für übersichtlichere Experiment-Konfiguration
    ppo_config = (PpoModelParamsBuilder(variant=model_variant,
                                        n_steps=n_steps,
                                        batch_size=batch_size,
                                        n_epochs=4,
                                        learning_rate=3e-4)
                  .with_exploration_settings(ent_coef=0.01, clip_range=0.2)
                  .build())

    # noinspection PyTypeChecker
    ppo_params.set(ppo_config)

    # --- TASK CONFIGURATION ---
    task_params = HyperparameterRegistry.get_store(TaskParams)
    task_config = (TaskParamsBuilder(core_env=core_env,
                                     transformer_labels=transformer_labels,
                                     max_transformations=max_transformations)
                   .with_rewards(success_bonus=1.0)
                   # .with_multi_step_logic(steps_per_episode=2) # Optional
                   .build())
    task_params.set(task_config)

    # --- RUNTIME CONFIGURATION ---
    runtime_params = HyperparameterRegistry.get_store(RuntimeParams)
    runtime_config = (RuntimeParamsBuilder(experiment_name="SB3_POC_IMAGE_OPTIMIZATION",
                                           run_name=run_name,
                                           total_training_steps=300_000,
                                           num_vector_envs=NUM_VECTOR_ENVS)
                      .with_random_seed(42)
                      .with_resource_settings(use_worker_pool=OPTIMIZE_FOR_MULTIPROCESSING,
                                              num_juror_workers=5,
                                              vec_env_cls="SubprocVecEnv" if OPTIMIZE_FOR_MULTIPROCESSING else "DummyVecEnv")
                      .with_evaluation(interval=n_steps * NUM_VECTOR_ENVS, visual_history=True)
                      .build())
    runtime_params.set(runtime_config)

    data_params = HyperparameterRegistry.get_store(DataParams)
    data_params.set({"dataset_id": dataset_id, "image_max_size": (384, 384)})

    trainer = StableBaselineTrainer()
    trainer.run_training(run_name=run_name)


if __name__ == '__main__':
    main()
