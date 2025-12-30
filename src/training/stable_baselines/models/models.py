from typing import Union

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import VecEnv

from training.stable_baselines.models.base_feature_extractor import ResNetFeatureExtractor, ResNet18FeatureExtractor


def create_resnet_extractor(extractorClass, feature_dim: int, freeze_backbone: bool = True) -> dict[str, type[ResNetFeatureExtractor] | dict[str, int | bool]]:
    return {
        "features_extractor_class": extractorClass,
        "features_extractor_kwargs": {
            "features_dim": feature_dim,
            "pretrained": True,
            "freeze_backbone": freeze_backbone
        },
        "normalize_images": False
    }


def create_ppo_with_resnet_model(vec_env: VecEnv,
                                 n_steps: int,
                                 batch_size: int,
                                 n_epochs: int,
                                 learning_rate: Union[float, Schedule] = 3e-4,
                                 feature_dim: int = 512
                                 ) -> PPO:
    return PPO("CnnPolicy",
               env=vec_env,
               n_steps=n_steps,
               learning_rate=learning_rate,
               batch_size=batch_size,
               n_epochs=n_epochs,
               policy_kwargs=create_resnet_extractor(ResNetFeatureExtractor, feature_dim),
               verbose=1)

def create_ppo_with_resnet18_model(vec_env: VecEnv,
                                 n_steps: int,
                                 batch_size: int,
                                 n_epochs: int,
                                 learning_rate: Union[float, Schedule] = 3e-4,
                                 feature_dim: int = 512,
                                 freeze_backbone: bool = True
                                 ) -> PPO:
    return PPO("CnnPolicy",
               env=vec_env,
               n_steps=n_steps,
               learning_rate=learning_rate,
               batch_size=batch_size,
               n_epochs=n_epochs,
               policy_kwargs=create_resnet_extractor(ResNet18FeatureExtractor, feature_dim, freeze_backbone),
               verbose=1)

def create_ppo_model_without_backbone(vec_env: VecEnv,
                                 n_steps: int,
                                 batch_size: int,
                                 n_epochs: int,
                                 learning_rate: Union[float, Schedule] = 3e-4,
                                 ) -> PPO:
    return PPO("CnnPolicy",
               env=vec_env,
               n_steps=n_steps,
               learning_rate=learning_rate,
               batch_size=batch_size,
               n_epochs=n_epochs,
               policy_kwargs={"normalize_images": False},
               verbose=1)

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