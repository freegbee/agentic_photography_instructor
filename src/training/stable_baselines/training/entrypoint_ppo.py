import os
import time

from torch import nn

from training.stable_baselines.rewards.reward_strategies import RewardStrategyEnum, SuccessBonusStrategyEnum
from utils.LoggingUtils import configure_logging

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

# Fix for MLflow system metrics crash on Windows
from training.stable_baselines.utils.utils import fix_psutil_disk_usage_on_windows
fix_psutil_disk_usage_on_windows()

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
from training.stable_baselines.models.model_factory import PpoModelFactory
from training.stable_baselines.training.trainer import StableBaselineTrainer
from transformer import SENSIBLE_TRANSFORMERS

configure_logging()


def main():
    opt_str = "MultiProc" if OPTIMIZE_FOR_MULTIPROCESSING else "SingleProc"

    # ========================================================================================
    # 1. EXPERIMENT DEFINITION (THE "WHAT")
    # Hier definieren wir die fachlichen Parameter des Experiments.
    # ========================================================================================
    experiment_name = "SB3_POC_IMAGE_OPTIMIZATION"
    run_description = "Flickr2k, HQ, Optuna Optimized, really correct Step Penalized, slightly adjusted values"

    # Daten & Umgebung
    dataset_id = "flickr2k_big_original_HQ_split_amd-win"
    image_size = (384, 384)
    core_env = WellDefinedEnvironment.IMAGE_OPTIMIZATION
    transformer_labels = SENSIBLE_TRANSFORMERS
    max_transformations = 10
    training_batch_size=64

    # Modell Konfiguration
    model_variant = PpoModelVariant.PPO_WITHOUT_BACKBONE
    learning_rate = 5.6115164153345e-05
    gamma=0.998
    ent_coef=0.035  # 5.3e-05
    reward_strategy = RewardStrategyEnum.STOP_ONLY_QUADRATIC
    success_bonus_strategy = SuccessBonusStrategyEnum.FIXED
    success_bonus = 1.0
    step_penalty = -0.002

    # ========================================================================================
    # 2. EXECUTION MODE (THE "HOW")
    # Hier steuern wir technische Parameter für Debugging vs. echtes Training.
    # ========================================================================================
    IS_DEBUG_RUN = False  # <--- HIER UMSCHALTEN: True für schnellen Test, False für Training

    if IS_DEBUG_RUN:
        print("\n!!! RUNNING IN DEBUG MODE (Short Rollouts, Fast Updates) !!!\n")
        target_rollout_size = 320  # Klein, aber durch batch_size teilbar
        batch_size = 32  # Kleine Batches für schnelle Updates
        total_training_steps = 2000  # Nur kurz anlaufen lassen
        n_epochs = 2  # Weniger Epochen für Speed
        run_name_prefix = f"DEBUG - {run_description}"
        store_models = False
    else:
        target_rollout_size = 4000  # Standard PPO Größe für stabiles Lernen
        batch_size = training_batch_size
        total_training_steps = 300_000
        n_epochs = 7
        run_name_prefix = run_description
        store_models = True

    # ========================================================================================
    # 3. CALCULATION & ASSEMBLY
    # Berechnung abgeleiteter Werte und Befüllen der Builder.
    # ========================================================================================

    # 3.1 Berechnungen
    n_steps = PpoModelParamsBuilder.calculate_n_steps(NUM_VECTOR_ENVS, target_rollout_size, batch_size=batch_size)
    rollout_size = n_steps * NUM_VECTOR_ENVS
    run_name = f"{time.strftime('%Y%m%d-%H%M%S')} - {run_name_prefix}, {model_variant.value} ({opt_str})"

    print(f"Configuration: Envs={NUM_VECTOR_ENVS}, n_steps={n_steps} (Total Rollout={rollout_size})")

    # 3.2 PPO Model Parameters
    ppo_params = HyperparameterRegistry.get_store(PpoModelParams)
    ppo_config = (PpoModelParamsBuilder(variant=model_variant,
                                        n_steps=n_steps,
                                        batch_size=batch_size,
                                        n_epochs=n_epochs,
                                        learning_rate=learning_rate)
                  .with_exploration_settings(ent_coef=ent_coef, clip_range=0.228)
                  .with_advantage_estimation(gamma=gamma, gae_lambda=0.965)
                  .with_net_arch(dict(pi=[256, 256], vf=[512, 512]))
                  .with_activation_fn(nn.ReLU)
                  .build())
    ppo_params.set(ppo_config)

    # 3.3 Task Parameters
    task_params = HyperparameterRegistry.get_store(TaskParams)
    task_config = (TaskParamsBuilder(core_env=core_env,
                                     transformer_labels=transformer_labels,
                                     max_transformations=max_transformations)
                   .with_rewards(strategy=reward_strategy, success_bonus_strategy=success_bonus_strategy, success_bonus=success_bonus, step_penalty=step_penalty)
                   .build())
    task_params.set(task_config)

    # 3.4 Runtime Parameters
    runtime_params = HyperparameterRegistry.get_store(RuntimeParams)
    runtime_config = (RuntimeParamsBuilder(experiment_name=experiment_name,
                                           run_name=run_name,
                                           total_training_steps=total_training_steps,
                                           num_vector_envs=NUM_VECTOR_ENVS)
                      .with_random_seed(42)
                      .with_resource_settings(use_worker_pool=OPTIMIZE_FOR_MULTIPROCESSING,
                                              num_juror_workers=5,
                                              vec_env_cls="SubprocVecEnv" if OPTIMIZE_FOR_MULTIPROCESSING else "DummyVecEnv")
                      # Evaluation immer genau nach einem vollen Rollout
                      .with_evaluation(interval=rollout_size, visual_history=True)
                      .with_model_storage(store_best_model=store_models, store_final_model=store_models)
                      .build())
    runtime_params.set(runtime_config)

    # 3.5 Data Parameters
    data_params = HyperparameterRegistry.get_store(DataParams)
    data_params.set({"dataset_id": dataset_id, "image_max_size": image_size})

    # ========================================================================================
    # 4. START
    # Berechnung abgeleiteter Werte und Befüllen der Builder.
    # ========================================================================================
    trainer = StableBaselineTrainer(model_factory=PpoModelFactory(),
                                    model_params_class=PpoModelParams)
    trainer.run_training(run_name=run_name)


if __name__ == '__main__':
    main()
