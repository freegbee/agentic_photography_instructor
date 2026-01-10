import math
from abc import ABC, abstractmethod
from enum import Enum


class RewardStrategyEnum(Enum):
    STOP_ONLY_PLAIN = "stop_only_plain"
    STOP_ONLY_QUADRATIC = "stop_only_quadratic"
    SCORE_DIFFERENCE = "score_difference"  # Standard: Delta + Bonus
    CLIPPED_DIFFERENCE = "clipped_difference"  # Delta geclippt, um Ausreißer zu vermeiden
    STEP_PENALIZED = "step_penalized"  # Delta + Bonus - Zeitstrafe


class SuccessBonusStrategyEnum(Enum):
    FIXED = "fixed"  # Einfacher, fixer Bonus
    SIGMOID = "sigmoid"  # Dynamischer Bonus basierend auf Bildqualität (Lob)


class AbstractSuccessBonusStrategy(ABC):
    def __init__(self, bonus_amount: float):
        self.bonus_amount = bonus_amount

    @abstractmethod
    def calculate(self, initial_score: float, new_score: float) -> float:
        pass


class FixedSuccessBonusStrategy(AbstractSuccessBonusStrategy):
    def calculate(self, initial_score: float, new_score: float) -> float:
        if initial_score is not None and new_score >= initial_score:
            return self.bonus_amount
        return 0.0


class SigmoidSuccessBonusStrategy(AbstractSuccessBonusStrategy):
    def calculate(self, initial_score: float, new_score: float) -> float:
        if initial_score is None or new_score < initial_score:
            return 0.0

        # Dynamische Skalierung des Bonus basierend auf der Qualität des Ausgangsbildes (Sigmoid).
        try:
            # Sigmoid centered at 5.0: f(x) = 1 / (1 + e^-(x-5))
            sigmoid_val = 1 / (1 + math.exp(-(initial_score - 5.0)))
            multiplier = 0.5 + sigmoid_val
            return self.bonus_amount * multiplier
        except OverflowError:
            return self.bonus_amount * (1.5 if initial_score > 5 else 0.5)


class AbstractRewardStrategy(ABC):
    """
    Basisklasse für Reward-Berechnungen.
    """

    def __init__(self, success_bonus_strategy: AbstractSuccessBonusStrategy, step_penalty: float = 0):
        self.success_bonus_strategy = success_bonus_strategy
        self.step_penalty = step_penalty

    def is_success(self, initial_score: float, new_score: float) -> bool:
        return initial_score is not None and new_score >= initial_score

    def calculate_success_bonus(self, initial_score: float, new_score: float) -> float:
        return self.success_bonus_strategy.calculate(initial_score, new_score)

    @abstractmethod
    def calculate(self,
                  transformer_label: str,
                  current_score: float,
                  new_score: float,
                  initial_score: float,
                  step_count: int,
                  max_steps: int) -> float:
        pass

class AbstractStopOnlyReward(AbstractRewardStrategy):

    @abstractmethod
    def _calc_reward(self, initial_score: float, new_score: float) -> float:
        pass

    def calculate(self,
                  transformer_label: str,
                  current_score: float,
                  new_score: float,
                  initial_score: float,
                  step_count: int,
                  max_steps: int) -> float:
        if transformer_label != 'STOP':
            return 0

        reward = self._calc_reward(new_score, initial_score)
        bonus = self.calculate_success_bonus(initial_score, new_score)
        penalty = self.step_penalty * step_count
        effective_reward = reward + bonus + penalty
        return effective_reward

class StopOnlyPlainReward(AbstractStopOnlyReward):
    """
    Reward wird nur am Ende zurückgegeben, wenn STOP gewählt wurde.
    Er ist einfach die Differenz aus new_score - initial_score, also die Gesamtverbesserung
    """
    def _calc_reward(self, initial_score: float, new_score: float) -> float:
        return new_score - initial_score


class StopOnlyQuadraticReward(AbstractStopOnlyReward):
    """
    Reward wird nur am Ende zurückgegeben, wenn STOP gewählt wurde.
    Er ist die Differenz der Quadrate: (new_score^2) - (initial_score^2). Belohnt höhere Scores überproportional.
    """
    def _calc_reward(self, initial_score: float, new_score: float) -> float:
        return (new_score ** 2) - (initial_score ** 2)


class ScoreDifferenceStrategy(AbstractRewardStrategy):
    """
    Der Klassiker: Reward ist die Verbesserung des Scores + Bonus bei Erfolg.
    """

    def calculate(self, transformer_label: str,
                  current_score: float,
                  new_score: float,
                  initial_score: float,
                  step_count: int,
                  max_steps: int) -> float:
        reward = new_score - current_score

        # Success Check
        reward += self.calculate_success_bonus(initial_score, new_score)

        return reward


class StepPenalizedStrategy(AbstractRewardStrategy):
    """
    Wie ScoreDifference, aber jeder Schritt kostet etwas, um den Agenten zur Eile zu treiben.
    """

    def __init__(self, success_bonus_strategy: AbstractSuccessBonusStrategy, step_penalty: float = -0.01):
        super().__init__(success_bonus_strategy, step_penalty)

    def calculate(self,
                  transformer_label: str,
                  current_score: float,
                  new_score: float,
                  initial_score: float,
                  step_count: int,
                  max_steps: int) -> float:
        reward = new_score - current_score

        reward += self.calculate_success_bonus(initial_score, new_score)

        # Zeitstrafe
        reward += self.step_penalty
        return reward


class ClippedDifferenceStrategy(AbstractRewardStrategy):
    """
    Verhindert, dass ein einzelner riesiger Sprung das Training destabilisiert.
    """

    def __init__(self, success_bonus_strategy: AbstractSuccessBonusStrategy, step_penalty: float = -0.01, min_clip: float = -0.5, max_clip: float = 0.5):
        super().__init__(success_bonus_strategy, step_penalty)
        self.min_clip = min_clip
        self.max_clip = max_clip

    def calculate(self,
                  transformer_label: str,
                  current_score: float,
                  new_score: float,
                  initial_score: float,
                  step_count: int,
                  max_steps: int) -> float:
        delta = new_score - current_score

        # Clipping des Deltas
        reward = max(self.min_clip, min(self.max_clip, delta))

        reward += self.calculate_success_bonus(initial_score, new_score)

        return reward
