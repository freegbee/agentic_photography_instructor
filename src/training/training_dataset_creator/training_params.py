from typing import TypedDict

from training.split_ratios import SplitRatios


class TransformPreprocessingParams(TypedDict):
    batch_size: int
    split: SplitRatios


class DataParams(TypedDict):
    dataset_id: str


class TrainingExecutionParams(TypedDict, total=False):
    experiment_name: str
    run_name: str
    use_local_juror: bool
    random_seed: int
