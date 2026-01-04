from enum import Enum


class PpoModelVariant(Enum):
    PPO_WITHOUT_BACKBONE = "PPO without Backbone"
    PPO_RESNET18_UNFROZEN = "PPO ResNet18 (Unfrozen)"
    PPO_RESNET18_FROZEN = "PPO ResNet18 (Frozen)"
    PPO_RESNET50_UNFROZEN = "PPO ResNet50 (Unfrozen)"
    PPO_RESNET50_FROZEN = "PPO ResNet50 (Frozen)"