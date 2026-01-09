from typing import Union

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
         :param variant: Die Variante des PPO-Modells (z.B. Architektur des Feature Extractors).
         :param n_steps: Anzahl der Schritte pro Environment, die gesammelt werden, bevor ein Update erfolgt.
         :param batch_size: Größe der Minibatches für das Gradienten-Update.
         :param n_epochs: Anzahl der Epochen, die über den gesammelten Buffer optimiert wird. Erst danach wird gelernt. Werte zwischen 1 (oberflächliches lernen) und 10 (overfitting-risiko). Für Bildlernen ist 4 gut.
         :param learning_rate: Die Lernrate (float) oder ein Lernraten-Schedule.
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

    def with_net_arch(self, net_arch: Union[list[int], dict[str, list[int]]]) -> "PpoModelParamsBuilder":
        self._params["net_arch"] = net_arch
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

    @staticmethod
    def calculate_n_steps(num_envs: int, target_total: int = 4000, batch_size: int = 100) -> int:
        """
        Berechnet n_steps so, dass die totale Anzahl Steps (n_steps * num_envs)
        nahe am target_total liegt UND durch batch_size teilbar ist (wichtig für PPO).
        """
        ideal_n = target_total // num_envs

        # Suche nach dem nächsten passenden Wert (aufwärts und abwärts)
        for i in range(1000):
            # Check up
            n = ideal_n + i
            if (n * num_envs) % batch_size == 0:
                return n
            # Check down
            n = ideal_n - i
            if n > 0 and (n * num_envs) % batch_size == 0:
                return n
        return ideal_n
