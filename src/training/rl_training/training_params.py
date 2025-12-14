from typing import TypedDict

from training.split_ratios import SplitRatios


class GeneralPreprocessingParams(TypedDict):
    batch_size: int
    random_seed: int


class ImagePreprocessingParams(TypedDict):
    batch_size: int
    resize_max_size: int


class TransformPreprocessingParams(TypedDict):
    batch_size: int
    transformer_names: list[str]
    use_random_transformer: bool
    split: SplitRatios


class DataParams(TypedDict):
    dataset_id: str


class TrainingExecutionParams(TypedDict, total=False):
    experiment_name: str
    run_name: str
    use_local_juror: bool
    random_seed: int
    # num_episodes: int
    # max_steps_per_episode: int
    # learning_rate: float
    # discount_factor: float
    # exploration_rate: float
    # exploration_decay: float
    # min_exploration_rate: float
