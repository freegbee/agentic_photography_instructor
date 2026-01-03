from typing import TypedDict, Tuple, Union

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class GeneralParams(TypedDict):
    success_bonus: float
    learning_rate: float
    transformer_labels: list[str]
    image_max_size: Tuple[int, int]
    vec_env_cls: Union[type[DummyVecEnv] | type[SubprocVecEnv] | None]
    use_worker_pool: bool  # Falls ein Juror Worker Pool genutzt werden soll
    num_juror_workers: int  # Wie viele im Pool sein sollen (z.B. Memory-Limitierung)
