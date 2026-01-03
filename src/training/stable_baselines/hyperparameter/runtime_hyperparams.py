from typing import TypedDict, Union

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class RuntimeParams(TypedDict):
    # Name des Experiments (für MLflow/Logging Gruppierung).
    experiment_name: str

    # Spezifischer Name dieses Trainingslaufs.
    run_name: str

    # Globaler Random Seed für Reproduzierbarkeit.
    random_seed: int

    # Gesamtzahl der Timesteps, die trainiert werden sollen.
    total_training_steps: int
    
    # Execution Environment

    # Anzahl der parallelen Environments (Vector Environments).
    num_vector_envs: int

    # Klasse für die Vektorisierung (DummyVecEnv für Single-Thread, SubprocVecEnv für Multi-Process).
    vec_env_cls: Union[type[DummyVecEnv] | type[SubprocVecEnv] | None]

    # Ob ein separater Worker-Pool für die Bewertung (Juror) genutzt werden soll (GPU-Entlastung).
    use_worker_pool: bool

    # Anzahl der Worker im Pool (falls use_worker_pool=True).
    num_juror_workers: int

    # Ob der Juror lokal im Environment-Prozess ausgeführt werden soll (falls kein Pool).
    use_local_juror: bool
    
    # Rendering & Logging

    # Modus für das Rendering während des Trainings (z.B. "human", "rgb_array", "skip").
    render_mode: str

    # Verzeichnis, in dem Renderings gespeichert werden.
    render_save_dir: str
    
    # Evaluation

    # Seed für die Evaluation-Environments.
    evaluation_seed: int

    # Intervall (in Timesteps), in dem die Evaluation durchgeführt wird.
    evaluation_interval: int

    # Ob die Policy während der Evaluation deterministisch handeln soll (kein Zufall).
    evaluation_deterministic: bool

    # Ob eine visuelle Historie der Änderungen erstellt werden soll.
    evaluation_visual_history: bool

    # Maximale Anzahl an Bildern in der visuellen Historie.
    evaluation_visual_history_max_images: int

    # Maximale Größe der Bilder in der Historie (für Speicheroptimierung).
    evaluation_visual_history_max_size: int

    # Render-Modus für die Evaluation.
    evaluation_render_mode: str

    # Speicherort für Evaluation-Renderings.
    evaluation_render_save_dir: str

    # Speicherort für die besten Modelle.
    evaluation_model_save_dir: str

    # Pfad für Evaluations-Logs.
    evaluation_log_path: str