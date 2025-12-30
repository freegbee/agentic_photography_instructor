from typing import Union

from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import VecEnv

from training.stable_baselines.models.base_feature_extractor import ResNetFeatureExtractor, ResNet18FeatureExtractor


def create_resnet_extractor(extractor_class, feature_dim: int, freeze_backbone: bool = True) -> dict[
    str, type[ResNetFeatureExtractor] | dict[str, int | bool]]:
    return {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {
            "features_dim": feature_dim,
            "pretrained": True,
            "freeze_backbone": freeze_backbone
        },
        "normalize_images": False
    }


def create_dqn_with_resnet_model(vec_env: VecEnv,
                                 learning_rate: Union[float, Schedule] = 1e-4,
                                 buffer_size: int = 100_000,
                                 batch_size: int = 32,
                                 learning_starts: int = 1_000,
                                 train_freq: int = 4,
                                 feature_dim: int = 512
                                 ) -> DQN:
    return DQN("CnnPolicy",
               env=vec_env,
               learning_rate=learning_rate,
               buffer_size=buffer_size,
               batch_size=batch_size,
               learning_starts=learning_starts,
               train_freq=train_freq,
               policy_kwargs=create_resnet_extractor(ResNet18FeatureExtractor, feature_dim),
               verbose=1)
