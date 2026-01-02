from typing import TypedDict, Optional

from training.split_ratios import SplitRatios


class GeneralPreprocessingParams(TypedDict):
    batch_size: int
    random_seed: int


class ImagePreprocessingParams(TypedDict, total=False):
    batch_size: int
    resize_max_size: int
    max_images: Optional[int]


class TransformPreprocessingParams(TypedDict, total=False):
    batch_size: int
    transformer_names: list[str]
    use_random_transformer: bool
    num_transformations: int  # Number of transformations to apply (default: 1)
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
