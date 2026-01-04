from enum import Enum


class DqnModelVariant(Enum):
    DQN_WITHOUT_BACKBONE = "DQN without Backbone"
    DQN_RESNET18_UNFROZEN = "DQN ResNet18 (Unfrozen)"
    DQN_RESNET18_FROZEN = "DQN ResNet18 (Frozen)"
    DQN_RESNET50_UNFROZEN = "DQN ResNet50 (Unfrozen)"
    DQN_RESNET50_FROZEN = "DQN ResNet50 (Frozen)"