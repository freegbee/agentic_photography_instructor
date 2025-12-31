from typing import TypedDict, Tuple

from training.stable_baselines.models.model_factory import PpoModelVariant


class DataParams(TypedDict):
    dataset_id: str


class TrainingParams(TypedDict):
    experiment_name: str
    run_name: str
    use_local_juror: bool
    random_seed: int
    ppo_model_variant: PpoModelVariant
    num_vector_envs: int
    mini_batch_size: int
    n_steps: int
    n_epochs: int
    max_transformations: int
    total_training_steps: int
    render_mode: str
    render_save_dir: str
    evaluation_seed: int
    evaluation_interval: int
    evaluation_deterministic: bool
    evaluation_visual_history: bool
    evaluation_visual_history_max_images: int
    evaluation_visual_history_max_size: int
    evaluation_render_mode: str
    evaluation_render_save_dir: str
    evaluation_model_save_dir: str
    evaluation_log_path: str


class GeneralParams(TypedDict):
    success_bonus: float
    learning_rate: float
    transformer_labels: list[str]
    image_max_size: Tuple[int, int]