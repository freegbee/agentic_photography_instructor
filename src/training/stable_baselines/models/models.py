from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from training.stable_baselines.models.base_feature_extractor import ResNetFeatureExtractor


def create_ppo_with_resnet_model(vec_env: VecEnv, feature_dim: int = 512) -> PPO:
    policy_kwargs = {
        "features_extractor_class": ResNetFeatureExtractor,
        "features_extractor_kwargs": {
            #
            "features_dim": feature_dim,
            "pretrained": True,
            "freeze_backbone": True,
        }
    }
    return PPO("CnnPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)