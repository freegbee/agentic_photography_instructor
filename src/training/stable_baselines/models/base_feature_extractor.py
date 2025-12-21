import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torchvision.models import resnet50
from torchvision import models


class ResNetFeatureExtractor(BaseFeaturesExtractor):
    """
        BaseFeaturesExtractor, das ein vortrainiertes ResNet50 als Feature-Backbone nutzt.
        Erwartet Eingaben im Tensorformat (B, C, H, W) mit Werten in [0,1].
        """

    def __init__(self, observation_space, features_dim: int = 512, pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__(observation_space, features_dim)
        backbone = resnet50(weights = models.ResNet50_Weights.DEFAULT if pretrained else None)
        # Entferne die finale fully-connected Schicht (keep conv+avgpool)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)  # Ausgabe: (B, 2048, 1, 1)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(2048, features_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # img: numpy array (H, W, C)

        # x: (B, C, H, W), Werte in [0,1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        h = self.backbone(x)  # (B, 2048, 1, 1)
        h = h.reshape(h.size(0), -1)  # (B, 2048)
        return self.proj(h)  # (B, features_dim)
