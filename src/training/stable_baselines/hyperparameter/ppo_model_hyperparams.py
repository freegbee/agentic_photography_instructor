from typing import TypedDict, Optional, List, Dict, Union, Type
import torch.nn as nn

from stable_baselines3.common.type_aliases import Schedule

from training.stable_baselines.models.model_variants import PpoModelVariant


class PpoModelParams(TypedDict):
    # Die Variante des PPO-Modells (z.B. mit/ohne Backbone, Frozen/Unfrozen).
    # Bestimmt die Architektur des Feature Extractors.
    ppo_model_variant: PpoModelVariant

    # Die Anzahl der Schritte, die pro Environment pro Update gesammelt werden.
    # Die Größe des Rollout Buffers ist n_steps * n_envs.
    n_steps: int

    # Die Größe der Minibatches für das Gradienten-Update.
    # Muss ein Teiler der Buffer-Größe (n_steps * n_envs) sein.
    batch_size: int

    # Anzahl der Epochen, die über den gesammelten Buffer optimiert wird.
    # Wie oft die gesammelten Daten für das Training wiederverwendet werden.
    n_epochs: int

    # Der Zeitplan für die Lernrate (Learning Rate Schedule).
    # Kann konstant sein oder über die Zeit abnehmen.
    model_learning_schedule: Schedule

    # Model behavior parameters

    # Entropy Coefficient: Steuert die Exploration. Ein höherer Wert zwingt den Agenten dazu,
    # "zufälligere" Aktionen auszuprobieren. (Standard: 0.0)
    ent_coef: Optional[float]

    # Clip Range: Begrenzt, wie stark sich die neue Policy von der alten unterscheiden darf.
    # Kern von PPO für Stabilität. (Standard: 0.2)
    clip_range: Optional[float]

    # Discount Factor: Bestimmt, wie wichtig zukünftige Belohnungen im Vergleich zu sofortigen sind.
    # (Standard: 0.99)
    gamma: Optional[float]

    # GAE Lambda: Steuert den Trade-off zwischen Bias und Varianz bei der Vorteilsschätzung
    # (Generalized Advantage Estimator). (Standard: 0.95)
    gae_lambda: Optional[float]

    # Max Grad Norm: Gradient Clipping. Verhindert "explodierende Gradienten",
    # die das Training destabilisieren können. (Standard: 0.5)
    max_grad_norm: Optional[float]

    # Netzwerkarchitektur von PPO "hinter" dem backbone
    net_arch: Optional[Union[List[int], Dict[str, List[int]]]]

    # Aktivierungsfunktion für die Hidden Layers (z.B. nn.ReLU, nn.Tanh)
    activation_fn: Optional[Type[nn.Module]]
