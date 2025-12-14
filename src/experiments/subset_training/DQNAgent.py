import logging
import random
from typing import List, Tuple, Callable, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

logger = logging.getLogger(__name__)


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class SimpleQNetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super(SimpleQNetwork, self).__init__()
        c, h, w = input_shape
        # a very small conv-net followed by linear layers
        # ensure h and w are divisible by 4 for the simple downsampling
        assert h % 4 == 0 and w % 4 == 0, "height and width must be divisible by 4"
        self.net = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (h // 4) * (w // 4), 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetFeatureQNetwork(nn.Module):
    """
    Verwende ResNet18 oder ResNet50 als Feature-Extractor und trainiere eigene lineare Layer oben drauf.

    - input_shape: Tuple (C,H,W). ResNet erwartet in der Regel 3 Kanäle; falls C != 3, wird eine 1x1-Conv-Mapping-Schicht verwendet.
    - backbone: 'resnet18' oder 'resnet50'
    - pretrained: ob vortrainierte Gewichte geladen werden sollen
    - freeze_backbone: wenn True, werden Backbone-Gewichte eingefroren (nur Kopf wird trainiert)

    Forward akzeptiert Tensoren im Bereich [0,255] (wie die restliche Implementierung) und skaliert intern durch 255.0.

    Hinweise zur Verwendung:
    Wichtige Hinweise zur Verwendung des ResNet-Backbones (zusammengefasst)
    - Input shape / Preprocessing:
        ResNet-Modelle, besonders wenn pretrained=True, erwarten typischerweise normalisierte Eingaben (ImageNet mean/std). In diesem Code wird das Bild intern nur durch 255.0 geteilt; wenn pretrained=True, stelle zusätzlich sicher, dass die gleiche Mean/Std-Normalisierung wie beim Pretraining (ImageNet) angewendet wird.
    - BatchNorm:
        Bei kleinen Batch-Größen können BatchNorm-Layer unstabil werden. Entweder größere Batches, Freeze/SyncBN oder BatchNorm-Alternativen einplanen.
    - Freeze / Fine-tune:
        Häufig: erst Kopf trainieren (backbone gefroren), dann schrittweise ganze Backbone feintunen.
    - Unterschiedliche Lernraten:
        übliche Praxis: Kopf (lr_head) größere LR, Backbone (lr_backbone) kleinere LR; per-parameter-group Optimizer einstellen.
    - Regularisierung:
        weight_decay für Backbone/Kopf, ggf. dropout im Kopf bei Overfitting.
    - Scheduler / Warmup:
        Lernratenplanung (z.B. CosineAnnealing, StepLR, Warmup) hilft Stabilität.
    - Gradient clipping:
        kann instabile Updates verhindern.
    - Bildgröße:
        ResNet kann mit adaptiver Pooling arbeiten, trotzdem empfiehlt sich konsistente Bildgröße (z. B. >=32).
    - Performance:
        ResNet50 ist deutlich schwerer als ResNet18 (Speicher/Speed). Abwägen nach Hardware/Ressourcen.
    - Weitere Parameter, die steuerbar sein sollten:
        freeze_backbone, pretrained, backbone-architektur, optimizer/param-groups, batch_size, input_norm mean/std, scheduler, weight_decay, dropout, warmup-steps.

    """

    def __init__(self, input_shape: Tuple[int, ...], n_actions: int,
                 backbone: str = 'resnet18', pretrained: bool = False, freeze_backbone: bool = True):
        super(ResNetFeatureQNetwork, self).__init__()
        c, h, w = input_shape

        # Keep simple check on image size - resnets work with many sizes because of adaptive pooling,
        # but some minimal size requirement is sensible (e.g. >= 32)
        assert h >= 32 and w >= 32, "height and width should be >= 32 for ResNet feature extraction"

        # If input channels != 3, map to 3 channels with a 1x1 conv so we can use standard ResNet.
        if c != 3:
            self.input_conv = nn.Conv2d(c, 3, kernel_size=1)
        else:
            self.input_conv = None

        # Choose backbone
        backbone = backbone.lower()
        if backbone == 'resnet18':
            # Use modern weights= API if available, otherwise fallback to pretrained
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                resnet = models.resnet18(weights=weights)
            except Exception:
                # Older torchvision: fallback to pretrained flag
                resnet = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet50':
            try:
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                resnet = models.resnet50(weights=weights)
            except Exception:
                resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone: {}. Choose 'resnet18' or 'resnet50'".format(backbone))

        # Get feature dimension from original fc and then replace fc with identity so model returns features
        # cast to int to satisfy static type checkers
        feature_dim = int(resnet.fc.in_features)
        resnet.fc = nn.Identity()
        self.backbone = resnet

        # Optionally freeze backbone parameters
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Small MLP head similar to SimpleQNetwork
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected in [0,255]
        x = x / 255.0
        if self.input_conv is not None:
            x = self.input_conv(x)
        feats = self.backbone(x)  # backbone returns flattened features because fc was replaced
        return self.head(feats)


class DQNAgent:
    """Very small DQN agent skeleton.

    - input states are numpy arrays shaped (C,H,W) and in [0,255] or floats.
    - actions are indices into the provided action list.

    Erweiterungen:
    - der Konstruktor akzeptiert jetzt optional einen network_constructor (Klasse oder callable),
      die ein nn.Module zurückgibt, und network_kwargs für Backbone-spezifische Parameter.

    Relevante Trainingsparameter für den Einsatz eines ResNet-Backbones (als Kommentar):
    - backbone (resnet18/resnet50): Wahl der Architektur; resnet50 bietet mehr Kapazität, ist aber langsamer.
    - pretrained (bool): ob mit ImageNet-Vortraining initialisiert werden soll. Hilfreich bei kleinen Datensätzen.
    - freeze_backbone (bool): ob das Backbone (teil-)eingefroren wird. Häufig zuerst nur Kopf trainieren, dann feintunen.
    - lr_head / lr_backbone: unterschiedliche Lernraten für Kopf und Backbone sind üblich (z.B. head lr höher).
    - weight_decay: Regularisierung für Backbone und Kopf.
    - batch_size: ResNet-Backbones erfordern üblicherweise größere Batchgrößen für stabile BatchNorm-Statistiken.
    - input normalization (mean/std): ResNets profitieren von der standardisierten Normalisierung (ImageNet mean/std) —
      stellen Sie sicher, dass Input-Skalierung und Normalisierung konsistent mit pretrained-Weights sind.
    - image size (H,W): ResNet kann verschiedene Größen verarbeiten, aber Training und Preprocessing sollten konsistent sein.
    - scheduler / warmup: Lernratenplanung (z.B. CosineAnnealing, StepLR) hilft bei stabiler Konvergenz.
    - gradient clipping: kann helfen, instabile Updates zu vermeiden.
    - dropout / regularization im Kopf: bei Überanpassung nützlich.
    - fine-tuning schedule: wie viele Epochs nur Kopf vs gesamtes Netz feintunen.

    Beispiel für network_constructor und network_kwargs zur Verwendung eines ResNet18-Backbones:
    ```
    agent = dqn.DQNAgent(action_space, state_shape, lr=1e-3,
                     network_constructor=dqn.ResNetFeatureQNetwork,
                     network_kwargs={'backbone': 'resnet18', 'pretrained': False, 'freeze_backbone': True, 'use_imagenet_norm': False})
    ```
    """

    def __init__(self, action_space: List[str], state_shape: Tuple[int, ...], lr: float = 1e-3,
                 lr_backbone: Optional[float] = None, lr_head: Optional[float] = None, weight_decay: float = 0.0,
                 gamma: float = 0.99, device=None,
                 network_constructor: Optional[Callable[..., nn.Module]] = None,
                 network_kwargs: Optional[Dict[str, object]] = None):
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.state_shape = state_shape
        self.network_constructor = network_constructor
        self.gamma = gamma
        self.device = device if device is not None else default_device()

        # Default network constructor: SimpleQNetwork
        if self.network_constructor is None:
            self.network_constructor = SimpleQNetwork
        if network_kwargs is None:
            network_kwargs = {}

        logger.info(f'Network Constructor: {network_constructor}')
        logger.info(f'Device: {self.device}')

        # Determine input normalization: if pretrained True -> use ImageNet mean/std
        # Allow overriding via network_kwargs: 'use_imagenet_norm' (bool), 'input_mean', 'input_std'
        pretrained_flag = bool(network_kwargs.get('pretrained', False))
        use_imagenet = bool(network_kwargs.get('use_imagenet_norm', pretrained_flag))
        custom_mean = network_kwargs.get('input_mean', None)
        custom_std = network_kwargs.get('input_std', None)

        if custom_mean is not None and custom_std is not None:
            self.input_mean = torch.tensor(custom_mean, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
            self.input_std = torch.tensor(custom_std, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
            self.apply_input_normalization = True
        elif use_imagenet:
            # ImageNet mean/std (range [0,1])
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.input_mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
            self.input_std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
            self.apply_input_normalization = True
        else:
            # only scale by 1/255. No channel-wise normalization.
            self.input_mean = None
            self.input_std = None
            self.apply_input_normalization = False

        # Remove agent-only keys from network_kwargs before passing to network constructor
        net_ctor_kwargs = {k: v for k, v in network_kwargs.items() if k not in ['use_imagenet_norm', 'input_mean', 'input_std']}

        # instantiate policy and target nets
        self.policy_net = self.network_constructor(state_shape, self.n_actions, **net_ctor_kwargs).to(self.device)
        self.target_net = self.network_constructor(state_shape, self.n_actions, **net_ctor_kwargs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Configure optimizer with per-parameter groups when possible.
        # Priority: if the network exposes `backbone` and `head` modules, create two groups with separate LRs.
        # Defaults: if lr_backbone or lr_head not provided, fall back to lr.
        backbone_lr = lr_backbone if lr_backbone is not None else lr
        head_lr = lr_head if lr_head is not None else lr

        # Collect trainable params from head and backbone if available
        head_params = []
        backbone_params = []
        try:
            if hasattr(self.policy_net, 'head'):
                head_params = [p for p in self.policy_net.head.parameters() if p.requires_grad]
        except Exception:
            head_params = []
        try:
            if hasattr(self.policy_net, 'backbone'):
                backbone_params = [p for p in self.policy_net.backbone.parameters() if p.requires_grad]
        except Exception:
            backbone_params = []

        # Remaining parameters that are neither in head nor backbone
        all_trainable = [p for p in self.policy_net.parameters() if p.requires_grad]
        head_ids = {id(p) for p in head_params}
        backbone_ids = {id(p) for p in backbone_params}
        remaining = [p for p in all_trainable if id(p) not in head_ids and id(p) not in backbone_ids]

        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr})
        if head_params:
            param_groups.append({'params': head_params, 'lr': head_lr})
        if remaining:
            # use base lr for any other parameters
            param_groups.append({'params': remaining, 'lr': lr})

        if len(param_groups) > 0:
            self.optimizer = optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
        else:
            # fallback: whole parameter set
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

        # epsilon for epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def _states_to_tensor(self, states, add_batch_dim: bool = False) -> torch.Tensor:
        """Utility: take numpy array or torch tensor of states and return a float tensor on self.device.

        - states: either numpy array or torch tensor with shape (C,H,W) or (N,C,H,W).
        - add_batch_dim: if True and states is (C,H,W), a leading batch dim is added.

        The function scales inputs from [0,255] to [0,1] (divides by 255.0) and applies optional
        channel-wise normalization if configured (self.apply_input_normalization).
        """
        # If numpy, convert to tensor
        if isinstance(states, np.ndarray):
            t = torch.tensor(states, dtype=torch.float32, device=self.device)
        elif isinstance(states, torch.Tensor):
            t = states.to(device=self.device, dtype=torch.float32)
        else:
            raise TypeError("states must be numpy array or torch Tensor")

        # Add batch dimension if needed
        if t.dim() == 3 and add_batch_dim:
            t = t.unsqueeze(0)

        # If values look like [0,255], scale them. If values already in [0,1], leave as-is.
        # Heuristic: if max>2 assume 0-255 range
        if t.max() > 2.0:
            t = t / 255.0

        # Ensure shape is (N,C,H,W)
        if t.dim() == 3:
            # still single sample (C,H,W) without batch dim
            t = t.unsqueeze(0)
        elif t.dim() != 4:
            raise ValueError(f"states must have 3 or 4 dims (C,H,W) or (N,C,H,W), got {t.shape}")

        # Optional channel-wise normalization
        if self.apply_input_normalization and self.input_mean is not None and self.input_std is not None:
            # If channel count differs (e.g. grayscale mapped earlier by network), broadcasting handles it.
            t = (t - self.input_mean) / self.input_std

        return t

    def select_action(self, state: np.ndarray) -> int:
        # state: (C,H,W) numpy
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        self.policy_net.eval()
        with torch.no_grad():
            t = self._states_to_tensor(state, add_batch_dim=True)
            q = self.policy_net(t)
            action = int(q.argmax(dim=1).item())
        return action

    def optimize_step(self, batch, target_update=False):
        # batch: tuple of (states, actions, rewards, next_states, dones)
        states, actions, rewards, next_states, dones = batch
        states_t = self._states_to_tensor(states, add_batch_dim=False)
        next_states_t = self._states_to_tensor(next_states, add_batch_dim=False)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            expected_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if target_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Epsilon decay should be called explicitly per environment step or episode.
        return float(loss.item())

    def action_to_string(self, action_idx: int) -> str:
        return self.action_space[action_idx]

    def decay_epsilon(self):
        """Decay epsilon, to be called per environment step or episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_checkpoint(self, filepath: str, **extra_info):
        """
        Save agent checkpoint including network weights and training state.

        Args:
            filepath: Path where to save the checkpoint
            **extra_info: Additional information to save (e.g., epoch, global_step)
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'action_space': self.action_space,
            'state_shape': self.state_shape,
            'gamma': self.gamma,
            **extra_info
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> dict:
        """
        Load agent checkpoint.

        Args:
            filepath: Path to the checkpoint file
            load_optimizer: Whether to load optimizer state (set False for inference)

        Returns:
            Dictionary with extra info from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']

        # Return extra info
        extra_info = {k: v for k, v in checkpoint.items()
                     if k not in ['policy_net_state_dict', 'target_net_state_dict',
                                  'optimizer_state_dict', 'epsilon', 'action_space',
                                  'state_shape', 'gamma']}

        return extra_info
