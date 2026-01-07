import gc
import os
import time

import torch
import mlflow
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from image_acquisition.acquisition_client.dummy_acquisition_client import DummyAcquisitionClient
from training import mlflow_helper
from training.hyperparameter_registry import HyperparameterRegistry
from training.stable_baselines.environment.welldefined_environments import WellDefinedEnvironment
from training.stable_baselines.hyperparameter.data_hyperparams import DataParams
from training.stable_baselines.hyperparameter.ppo_model_hyperparams import PpoModelParams
from training.stable_baselines.hyperparameter.ppo_model_hyperparams_builder import PpoModelParamsBuilder
from training.stable_baselines.hyperparameter.runtime_hyperparams import RuntimeParams
from training.stable_baselines.utils.utils import fix_psutil_disk_usage_on_windows
from training.stable_baselines.callbacks.optuna_pruning_callback import OptunaPruningCallback
from training.stable_baselines.hyperparameter.runtime_params_builder import RuntimeParamsBuilder
from training.stable_baselines.hyperparameter.task_hyperparams import TaskParams
from training.stable_baselines.hyperparameter.task_params_builder import TaskParamsBuilder
from training.stable_baselines.models.model_factory import PpoModelFactory
from training.stable_baselines.models.model_variants import PpoModelVariant
from training.stable_baselines.training.trainer import StableBaselineTrainer
from transformer import SENSIBLE_TRANSFORMERS
# Importiere deine bestehenden Klassen
from utils.LoggingUtils import configure_logging

# Logging konfigurieren
configure_logging()
fix_psutil_disk_usage_on_windows()

# Globale Konstanten für die Optimierung
N_TRIALS = 50  # Wie viele Versuche soll Optuna machen?
N_STARTUP_TRIALS = 5  # Die ersten 5 Versuche nicht prunen (um Baseline zu haben)
N_EVALUATIONS = 5  # Wie oft pro Training soll evaluiert werden?
TOTAL_TIMESTEPS_PER_TRIAL = 50_000  # Kürzere Trainingszeit für Optimierung (statt 300k)
NUM_VECTOR_ENVS = 4

# Technischer Test:
# Für den Testlauf:
# N_TRIALS = 2
# N_STARTUP_TRIALS = 1
# TOTAL_TIMESTEPS_PER_TRIAL = 2048 # Sehr kurz, nur um zu sehen ob es durchläuft


def objective(trial: optuna.Trial):
    """
    Die Optimierungs-Funktion für Optuna.
    """

    # ==========================================
    # Perforance
    # ==========================================
    dummy_acquisition_client = DummyAcquisitionClient()

    # ==========================================
    # 1. HYPERPARAMETER SAMPLING
    # ==========================================

    # Learning Rate: Logarithmische Skala ist hier wichtig
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # PPO Spezifika
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    # gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    # gamma = trial.suggest_float("gamma", 0.9, 0.9999)

    # Batch Size & Steps
    # WICHTIG: PPO mag es, wenn batch_size ein Teiler von (n_steps * n_envs) ist.
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # Wir wählen n_steps so, dass der Rollout Buffer ein Vielfaches der Batch Size ist
    # n_steps ist die Anzahl der Schritte PRO Environment
    # Wir probieren Faktoren für n_steps: 128 bis 2048
    n_steps_factor = trial.suggest_int("n_steps_factor", 2, 32)
    n_steps = batch_size * n_steps_factor // NUM_VECTOR_ENVS

    # Fallback, falls n_steps zu klein wird durch die Division
    if n_steps < 16:
        n_steps = 16

    # Memory Constraint: Max 512 steps um OOM zu verhindern
    if n_steps > 512:
        n_steps = 512

    # Sicherstellen, dass n_steps * NUM_VECTOR_ENVS durch batch_size teilbar ist (für SB3 PPO Logik)
    while (n_steps * NUM_VECTOR_ENVS) % batch_size != 0:
        n_steps -= 1

    # ==========================================
    # 2. SETUP (Ähnlich wie in entrypoint_ppo.py)
    # ==========================================

    experiment_name = "OPTUNA_PPO_OPTIMIZATION"
    run_name = f"trial_{trial.number}_{int(time.time())}"

    # PPO Params setzen
    ppo_params = HyperparameterRegistry.get_store(PpoModelParams)
    ppo_config = (PpoModelParamsBuilder(variant=PpoModelVariant.PPO_WITHOUT_BACKBONE,
                                        n_steps=n_steps,
                                        batch_size=batch_size,
                                        n_epochs=n_epochs,
                                        learning_rate=learning_rate)
                  .with_exploration_settings(ent_coef=ent_coef, clip_range=clip_range)
                  # Wir können hier auch gamma und gae_lambda setzen, falls dein Builder das unterstützt
                  # .with_gae_settings(gamma=gamma, gae_lambda=gae_lambda)
                  .build())

    # Falls dein Builder gamma/gae_lambda noch nicht explizit hat,
    # müsstest du sie ggf. manuell ins Dictionary patchen oder den Builder erweitern.
    # Hier nehmen wir an, sie sind Teil der Config oder Standard.
    ppo_params.set(ppo_config)

    # Task Params
    task_params = HyperparameterRegistry.get_store(TaskParams)
    task_config = (TaskParamsBuilder(core_env=WellDefinedEnvironment.IMAGE_OPTIMIZATION,
                                     transformer_labels=SENSIBLE_TRANSFORMERS,
                                     max_transformations=10)
                   .with_rewards(success_bonus=1.0)
                   .build())
    task_params.set(task_config)

    # Runtime Params
    # WICHTIG: Wir nutzen hier DummyVecEnv für Stabilität im Trial oder Subproc wenn es schnell sein muss
    # Für Optuna ist oft Single-Threaded pro Trial besser, wenn man viele Trials parallel laufen lässt,
    # aber hier lassen wir die Envs parallel laufen und Optuna sequenziell.
    runtime_params = HyperparameterRegistry.get_store(RuntimeParams)
    runtime_config = (RuntimeParamsBuilder(experiment_name=experiment_name,
                                           run_name=run_name,
                                           total_training_steps=TOTAL_TIMESTEPS_PER_TRIAL,
                                           num_vector_envs=NUM_VECTOR_ENVS)
                      .with_resource_settings(use_worker_pool=True,
                                              num_juror_workers=4,
                                              vec_env_cls="SubprocVecEnv")
                      .with_evaluation(interval=TOTAL_TIMESTEPS_PER_TRIAL // N_EVALUATIONS,
                                       visual_history=False)
                      .with_model_storage(store_best_model=False, store_final_model=False)  # Sparen I/O
                      .build())
    runtime_params.set(runtime_config)

    # Data Params
    data_params = HyperparameterRegistry.get_store(DataParams)
    data_params.set({"dataset_id": "flickr2k_big_original_HQ_split_amd-win",
                     "image_max_size": (384, 384)})

    # ==========================================
    # 3. TRAINING & PRUNING
    # ==========================================

    trainer = None
    try:
        # Pruning Callback erstellen
        pruning_callback = OptunaPruningCallback(trial)

        trainer = StableBaselineTrainer(model_factory=PpoModelFactory(),
                                        model_params_class=PpoModelParams,
                                        additional_callbacks=[pruning_callback],
                                        acquisition_client=dummy_acquisition_client)

        # Start Training
        trainer.run_training(run_name=run_name)

        # Den besten Reward vom Trainer abrufen (wird in _train_impl gesetzt)
        return trainer.best_mean_reward

    except optuna.exceptions.TrialPruned:
        # Wenn der Trial abgebrochen wird, müssen wir sicherstellen, dass der MLflow Run beendet wird,
        # damit der nächste Trial einen neuen Run starten kann.
        mlflow_helper.end_run()
        # Wichtig: Diese Exception muss durchgereicht werden, damit Optuna weiß, dass gepruned wurde
        raise
    except Exception as e:
        mlflow_helper.end_run()
        print(f"Trial failed with error: {e}")
        # Bei Fehlern (z.B. OOM) geben wir einen sehr schlechten Wert zurück
        return -1000.0
    finally:
        # ==========================================
        # 4. CLEANUP (Memory Leaks verhindern)
        # ==========================================
        if trainer is not None:
            del trainer
        
        # Zwinge Python, ungenutzte Objekte freizugeben
        gc.collect()
        
        # Leere den PyTorch GPU Cache, damit VRAM für den nächsten Trial frei wird
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Fix für Windows Multiprocessing
    if os.name == 'nt':
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Studie erstellen
    study = optuna.create_study(
        study_name="ppo_optimization_flickr",
        direction="maximize",  # Wir wollen den Reward maximieren
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=TOTAL_TIMESTEPS_PER_TRIAL // 3)
    )

    print(f"Start Optuna Optimization with {N_TRIALS} trials...")
    study.optimize(objective, n_trials=N_TRIALS)

    print("--------------------------------------------------")
    print("Best hyperparameters found:")
    print(study.best_params)
    print(f"Best Reward: {study.best_value}")
    print("--------------------------------------------------")

    # Speichern der besten Parameter in eine Datei
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"optimize_ppo_{study.study_name}_{timestamp}.optuna.txt"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    with open(output_path, "w") as f:
        f.write(f"# Best Reward: {study.best_value}\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")

    print(f"Best hyperparameters saved to {output_path}")
