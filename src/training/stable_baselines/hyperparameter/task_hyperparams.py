from typing import TypedDict

from training.stable_baselines.environment.welldefined_environments import WellDefinedEnvironment
from training.stable_baselines.rewards.reward_strategies import RewardStrategyEnum


class TaskParams(TypedDict):
    # Das Kern-Environment, das trainiert wird (z.B. Image Optimization, Restoration).
    # Definiert die grundlegende Logik und Belohnungsstruktur.
    core_env: WellDefinedEnvironment

    # Liste der verfügbaren Aktionen (Transformer), die der Agent wählen kann.
    transformer_labels: list[str]

    # Maximale Anzahl an Transformationen pro Episode, bevor diese abgebrochen wird.
    max_transformations: int

    # Bonus-Reward, der vergeben wird, wenn ein Zielzustand erreicht wird (falls definiert).
    success_bonus: float

    # Bestrafung für jeden einzelnen schritt
    step_penalty: float

    # Strategy zur Berechnung des Rewards
    reward_strategy: RewardStrategyEnum
    
    # Multi-step wrapper parameters

    # Aktiviert den Multi-Step Wrapper, bei dem der Agent mehrere Aktionen hintereinander ausführt,
    # bevor er eine neue Beobachtung erhält.
    use_multi_step_wrapper: bool

    # Anzahl der Schritte pro Episode (relevant für Multi-Step).
    steps_per_episode: int

    # Ob Zwischenschritte im Multi-Step-Prozess belohnt werden sollen.
    multi_step_intermediate_reward: bool

    # Ob Reward Shaping für Zwischenschritte angewendet werden soll.
    multi_step_reward_shaping: bool