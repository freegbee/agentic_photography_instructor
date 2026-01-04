from typing import Type, Union

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv

from training.stable_baselines.hyperparameter.dqn_model_hyperparams import DqnModelParams
from training.stable_baselines.models.base_feature_extractor import ResNetFeatureExtractor, ResNet18FeatureExtractor
from training.stable_baselines.models.dqn_model_variants import DqnModelVariant
from training.stable_baselines.models.model_factory import AbstractModelFactory


class DqnModelFactory(AbstractModelFactory[DqnModelParams]):
    def create_model(self, vec_env: VecEnv, params: DqnModelParams) -> DQN:
        # Gemeinsame Parameter extrahieren
        kwargs = {
            "buffer_size": params["buffer_size"],
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "learning_starts": params["learning_starts"],
            "target_update_interval": params["target_update_interval"],
            "train_freq": params["train_freq"],
            "gradient_steps": params["gradient_steps"],
            "exploration_fraction": params["exploration_fraction"],
            "exploration_initial_eps": params["exploration_initial_eps"],
            "exploration_final_eps": params["exploration_final_eps"],
            "gamma": params.get("gamma", 0.99),
            "verbose": 1
        }

        match params["dqn_model_variant"]:
            case DqnModelVariant.DQN_WITHOUT_BACKBONE:
                return DQN("CnnPolicy", env=vec_env, policy_kwargs={"normalize_images": False}, **kwargs)

            case DqnModelVariant.DQN_RESNET18_UNFROZEN:
                return self._create_resnet_model(vec_env, False, ResNet18FeatureExtractor, **kwargs)

            case DqnModelVariant.DQN_RESNET18_FROZEN:
                return self._create_resnet_model(vec_env, True, ResNet18FeatureExtractor, **kwargs)

            case DqnModelVariant.DQN_RESNET50_UNFROZEN:
                return self._create_resnet_model(vec_env, False, ResNetFeatureExtractor, **kwargs)

            case DqnModelVariant.DQN_RESNET50_FROZEN:
                return self._create_resnet_model(vec_env, True, ResNetFeatureExtractor, **kwargs)

            case _:
                raise ValueError(f"Unknown DQN model variant: {params['dqn_model_variant']}")

    def _create_resnet_model(self,
                             vec_env: VecEnv,
                             freeze_backbone: bool,
                             extractor_class: Type[Union[ResNetFeatureExtractor, ResNet18FeatureExtractor]],
                             **kwargs) -> DQN:
        
        # Feature Dim ist bei DQN oft kleiner oder gleich, aber ResNet liefert 512.
        # DQN hat keinen separaten Actor/Critic, sondern ein Q-Netz.
        policy_kwargs = {
            "features_extractor_class": extractor_class,
            "features_extractor_kwargs": {
                "features_dim": 512,
                "pretrained": True,
                "freeze_backbone": freeze_backbone
            },
            "normalize_images": False
        }

        return DQN("CnnPolicy",
                   env=vec_env,
                   policy_kwargs=policy_kwargs,
                   **kwargs)