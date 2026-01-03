from typing import Union, Optional

from stable_baselines3.common.type_aliases import Schedule

from training.stable_baselines.hyperparameter.ppo_model_hyperparams import PpoModelParams
from training.stable_baselines.models.learning_rate_schedules import linear_schedule
from training.stable_baselines.models.model_variants import PpoModelVariant


class PpoModelParamsBuilder:
    """
    Builder für PpoModelParams.
    Erzwingt die Eingabe der wichtigsten Parameter und erlaubt die gruppierte Konfiguration
    von optionalen Hyperparametern.
    """

    def __init__(self,
                 variant: PpoModelVariant,
                 n_steps: int,
                 batch_size: int,
                 n_epochs: int,
                 learning_rate: Union[float, Schedule]):
        """
        Initialisiert den Builder mit den zwingend erforderlichen Parametern.
        """
        # Falls learning_rate ein float ist, wandeln wir es direkt in einen Schedule um,
        # da PpoModelParams einen Schedule erwartet.
        if isinstance(learning_rate, float) or isinstance(learning_rate, int):
            lr_schedule = linear_schedule(learning_rate)
        else:
            lr_schedule = learning_rate

        # noinspection PyTypeChecker
        self._params: PpoModelParams = {
            "ppo_model_variant": variant,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "model_learning_schedule": lr_schedule,
        }
        self._set_default_values()

    def _set_default_values(self):
        """
        Setzt explizit die Standardwerte (Stable Baselines 3 Defaults),
        damit diese im Parameter-Set dokumentiert sind.
        """
        self._params["ent_coef"] = 0.0
        self._params["clip_range"] = 0.2
        self._params["gamma"] = 0.99
        self._params["gae_lambda"] = 0.95
        self._params["max_grad_norm"] = 0.5

    def with_exploration_settings(self, ent_coef: float, clip_range: float) -> "PpoModelParamsBuilder":
        """
        Setzt Parameter, die die Exploration und Stabilität der Policy-Updates steuern.
        :param ent_coef: Entropy Coefficient (höher = mehr Zufall/Exploration).
        :param clip_range: PPO Clipping (niedriger = stabilere, aber langsamere Updates).
        """
        self._params["ent_coef"] = ent_coef
        self._params["clip_range"] = clip_range
        return self

    def with_advantage_estimation(self, gamma: float, gae_lambda: float) -> "PpoModelParamsBuilder":
        """
        Setzt Parameter für die Berechnung der Belohnungsvorteile (Advantage).
        :param gamma: Discount Factor (Wichtigkeit zukünftiger Rewards).
        :param gae_lambda: Bias-Variance Trade-off für GAE.
        """
        self._params["gamma"] = gamma
        self._params["gae_lambda"] = gae_lambda
        return self

    def with_gradient_control(self, max_grad_norm: float) -> "PpoModelParamsBuilder":
        """
        Setzt Parameter zur Stabilisierung der Gradienten.
        :param max_grad_norm: Clipping-Wert für Gradienten.
        """
        self._params["max_grad_norm"] = max_grad_norm
        return self

    def build(self) -> PpoModelParams:
        """
        Gibt das fertige TypeDict zurück.
        """
        # Validierung könnte hier erweitert werden (z.B. batch_size Teiler Check)
        if (self._params["n_steps"] * 1) % self._params["batch_size"] != 0:
            # Hinweis: Wir kennen num_envs hier nicht zwingend, daher nur Warnung oder einfacher Check
            pass
        return self._params