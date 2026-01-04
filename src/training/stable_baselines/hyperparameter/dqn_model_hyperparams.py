from typing import TypedDict, Optional, Union, Tuple

from stable_baselines3.common.type_aliases import Schedule

from training.stable_baselines.models.dqn_model_variants import DqnModelVariant


class DqnModelParams(TypedDict):
    dqn_model_variant: DqnModelVariant

    # Größe des Replay Buffers (Anzahl der gespeicherten Transitions).
    buffer_size: int

    # Wie viele Schritte im Environment gesammelt werden, bevor ein Gradienten-Update erfolgt.
    # (train_freq)
    train_freq: Union[int, Tuple[int, str]]

    # Wie viele Gradienten-Steps pro train_freq ausgeführt werden.
    gradient_steps: int

    # Minibatch-Größe für das Update.
    batch_size: int

    # Lernrate (oder Schedule).
    learning_rate: Union[float, Schedule]

    # Ab wann das Training beginnt (um den Buffer erst etwas zu füllen).
    learning_starts: int

    # Wie oft (in Steps) das Target-Network aktualisiert wird.
    target_update_interval: int

    # Exploration Fraction: Anteil der total_timesteps, über den epsilon von initial auf final reduziert wird.
    exploration_fraction: float
    exploration_initial_eps: float
    exploration_final_eps: float

    gamma: float