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
from training.stable_baselines.models.model_factory import PpoModelVariant
from training.stable_baselines.training.hyper_params import TrainingParams, DataParams, GeneralParams
from training.stable_baselines.training.trainer import StableBaselineTrainer
from transformer import POC_MULTI_ONE_STEP_TRANSFORMERS, SENSIBLE_TRANSFORMERS
from utils.LoggingUtils import configure_logging
from utils.Registries import TRANSFORMER_REGISTRY


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

    # --- EXPERIMENT CONFIGURATION SWITCH ---
    training_modus = "Optimization"
    # ---------------------------------------

    model_variant = PpoModelVariant.PPO_WITHOUT_BACKBONE

    opt_str = "MultiProc" if OPTIMIZE_FOR_MULTIPROCESSING else "SingleProc"

    # Setup basierend auf Modus
    if training_modus == "Optimization":
        # Beispiel für das neue Setup
        run_name_prefix = "Landscapes, Multi-Optimization, Sensible"
        core_env = WellDefinedEnvironment.IMAGE_OPTIMIZATION
        transformer_labels = SENSIBLE_TRANSFORMERS
        dataset_id = "twenty_original_split_amd-win"
        max_transformations = 5
        # core_env_cls_name = "ImageTransformEnvVariant"
    else:
        # Default Setup
        run_name_prefix = "Landscapes, One-of-Six, Juror Scores"
        source_transformer_labels = POC_MULTI_ONE_STEP_TRANSFORMERS
        core_env = WellDefinedEnvironment.IMAGE_DEDEGRATION
        transformer_labels = [TRANSFORMER_REGISTRY.get(l).get_reverse_transformer_label() for l in
                              source_transformer_labels]
        dataset_id = "lhq_landscapes_multi_one_step_actions_amd-win"
        max_transformations = 1
        # core_env_cls_name = "ImageTransformEnv"

    run_name = f"{time.strftime('%Y%m%d-%H%M%S')} - {run_name_prefix}, {model_variant.value} ({opt_str})"

    # Dynamische Berechnung von n_steps für konstante Batch-Größe (~4000)
    target_rollout_size = 4000
    n_steps = calculate_n_steps(NUM_VECTOR_ENVS, target_rollout_size, batch_size=100)

    print(f"Configuration: Envs={NUM_VECTOR_ENVS}, n_steps={n_steps} (Total Rollout={n_steps * NUM_VECTOR_ENVS})")

    general_params = HyperparameterRegistry.get_store(GeneralParams)
    general_params.set({
        "success_bonus": 1.0,
        "learning_rate": 3e-4,
        "transformer_labels": transformer_labels,
        "image_max_size": (384, 384),
        "vec_env_cls": "SubprocVecEnv" if OPTIMIZE_FOR_MULTIPROCESSING else "DummyVecEnv",
        "use_worker_pool": True if OPTIMIZE_FOR_MULTIPROCESSING else False,
        "num_juror_workers": 5  # NEU: Anzahl GPU-Worker (VRAM Limit)
    })

    training_params = HyperparameterRegistry.get_store(TrainingParams)
    training_params.set({
        "experiment_name": "SB3_POC_IMAGE_OPTIMIZATION",
        "run_name": run_name,
        "use_local_juror": True,
        "random_seed": 42,
        "core_env": core_env,
        "ppo_model_variant": model_variant,
        "num_vector_envs": NUM_VECTOR_ENVS,
        "n_steps": n_steps,
        "mini_batch_size": 100,  # (n_steps * num_vector_env) % mini_batch_size == 0, also
        "n_epochs": 4,
        "max_transformations": max_transformations,
        "total_training_steps": 16_000,
        "render_mode": "skip",  # "save",
        "render_save_dir": "./renders/",

        # === MULTI-STEP WRAPPER CONFIGURATION ===
        "use_multi_step_wrapper": False,  # Enable multi-step wrapper
        "steps_per_episode": 2,  # Agent must take 2 actions per episode
        "multi_step_intermediate_reward": False,  # No reward for intermediate steps (default)
        "multi_step_reward_shaping": False,  # No shaped rewards (default)
        # =========================================

        # evaluation parameters
        "evaluation_seed": 67,
        "evaluation_interval": 4000,  # num_vector_envs * n_steps -> Nach jedem Rollout validieren
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
    data_params.set({"dataset_id": dataset_id})

    trainer = StableBaselineTrainer()
    trainer.run_training(run_name=run_name)


if __name__ == '__main__':
    main()
