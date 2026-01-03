import os
import time

from utils.LoggingUtils import configure_logging

# Konfiguration: Auch für DQN nutzen wir Multiprocessing, da unser Environment (Juror + Bild-Trafo)
# rechenintensiv ist. Wir wollen die GPU mit Batches füttern, statt sie warten zu lassen.
OPTIMIZE_FOR_MULTIPROCESSING = True
NUM_VECTOR_ENVS = 8 if OPTIMIZE_FOR_MULTIPROCESSING else 1

if OPTIMIZE_FOR_MULTIPROCESSING:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Fix for MLflow system metrics crash on Windows (psutil disk_usage)
if os.name == 'nt':
    try:
        import psutil
        # Monkey-patch psutil.disk_usage to suppress SystemError (bad format char)
        _original_disk_usage = psutil.disk_usage

        class _DummyUsage:
            total = 0; used = 0; free = 0; percent = 0

        def _robust_disk_usage(path):
            try:
                return _original_disk_usage(path)
            except Exception:
                return _DummyUsage()

        psutil.disk_usage = _robust_disk_usage
    except ImportError:
        pass

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

    dataset_id = "lhq_landscapes_original_split_amd-win"
    # dataset_id = "twenty_original_split_amd-win"
    image_size = (384, 384)
    # Für debug wird 1/10 verwendet. Definiert die Grösse des replay buffers
    # Siehe training.stable_baselines.hyperparameter.dqn_params_builder.DqnParamsBuilder.calculate_buffer_size
    # WICHTIG: Das ist der GESAMT-Speicher für den Buffer. Nicht durch NUM_VECTOR_ENVS teilen!
    # Der Buffer ist zentralisiert. Mehr Envs füllen ihn nur schneller.
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
        store_models = False
    else:
        buffer_size = DqnParamsBuilder.calculate_buffer_size(image_size, target_memory_mb=target_memory_mb)
        batch_size = 32
        learning_starts = 1000
        total_training_steps = 300_000
        evaluation_interval = 5000
        run_name_prefix = run_description
        store_models = True

    # ========================================================================================
    # 3. CALCULATION & ASSEMBLY
    # Berechnung abgeleiteter Werte und Befüllen der Builder.
    # ========================================================================================
    print(f"Calculated buffer_size: {buffer_size} transitions based on image size {image_size}")
    opt_str = "MultiProc" if OPTIMIZE_FOR_MULTIPROCESSING else "SingleProc"
    run_name = f"{time.strftime('%Y%m%d-%H%M%S')} - {run_name_prefix}, {model_variant.value} ({opt_str})"

    # 3.1 DQN Model Parameters
    dqn_params = HyperparameterRegistry.get_store(DqnModelParams)
    dqn_config = (DqnParamsBuilder(variant=model_variant,
                                   buffer_size=buffer_size,
                                   batch_size=batch_size,
                                   learning_rate=learning_rate)
                  # train_freq=1: Update nach jedem Schritt (wichtig bei kurzen Episoden von nur 10 Steps)
                  # gradient_steps: Da wir NUM_VECTOR_ENVS Schritte gleichzeitig sammeln, müssen wir auch
                  # entsprechend öfter trainieren, um das Verhältnis beizubehalten.
                  .with_training_schedule(learning_starts=learning_starts,
                                          train_freq=1,
                                          gradient_steps=1 * NUM_VECTOR_ENVS,
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
                      .with_resource_settings(use_worker_pool=OPTIMIZE_FOR_MULTIPROCESSING,
                                              num_juror_workers=5 if OPTIMIZE_FOR_MULTIPROCESSING else 0,
                                              vec_env_cls="SubprocVecEnv" if OPTIMIZE_FOR_MULTIPROCESSING else "DummyVecEnv",
                                              # WICHTIG: Bei Multiprocessing Juror NICHT lokal laden, sonst explodiert der RAM (8x Modell)
                                              use_local_juror=not OPTIMIZE_FOR_MULTIPROCESSING)
                      .with_evaluation(interval=evaluation_interval, visual_history=True)
                      .with_model_storage(store_best_model=store_models, store_final_model=store_models)
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