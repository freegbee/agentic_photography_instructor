from typing import TypedDict, Tuple


class DataParams(TypedDict):
    dataset_id: str


class TrainingParams(TypedDict):
    experiment_name: str
    run_name: str
    use_local_juror: bool
    random_seed: int
    num_vector_envs: int
    max_transformations: int
    total_training_steps: int

class GeneralParams(TypedDict):
    success_bonus: float
    transformer_labels: list[str]
    image_max_size: Tuple[int, int]