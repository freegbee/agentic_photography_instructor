from enum import Enum
from typing import Union, Type

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import VecEnv

from training.stable_baselines.models.base_feature_extractor import ResNetFeatureExtractor, ResNet18FeatureExtractor


class PpoModelVariant(Enum):
    PPO_WITHOUT_BACKBONE = "PPO without Backbone"
    PPO_RESNET18_UNFROZEN = "PPO ResNet18 (Unfrozen)"
    PPO_RESNET18_FROZEN = "PPO ResNet18 (Frozen)"
    PPO_RESNET50_UNFROZEN = "PPO ResNet50 (Unfrozen)"
    PPO_RESNET50_FROZEN = "PPO ResNet50 (Frozen)"


class PpoModelFactory:
    def __init__(self, model_variant: Union[PpoModelVariant, str]):
        if isinstance(model_variant, str):
            self.model_variant = PpoModelVariant(model_variant)
        else:
            self.model_variant = model_variant

    def create_model(self,
                     vec_env: VecEnv,
                     n_steps: int,
                     batch_size: int,
                     n_epochs: int,
                     learning_rate: Union[float, Schedule] = 3e-4,
                     **kwargs) -> PPO:
        match self.model_variant:
            case PpoModelVariant.PPO_WITHOUT_BACKBONE:
                return self._create_ppo_without_backbone(vec_env, n_steps, batch_size, n_epochs, learning_rate,
                                                         **kwargs)
            case PpoModelVariant.PPO_RESNET18_UNFROZEN:
                return self._create_resnet_model(vec_env, n_steps, batch_size, n_epochs, False,
                                                 ResNet18FeatureExtractor, learning_rate, **kwargs)
            case PpoModelVariant.PPO_RESNET18_FROZEN:
                return self._create_resnet_model(vec_env, n_steps, batch_size, n_epochs, True, ResNet18FeatureExtractor,
                                                 learning_rate, **kwargs)
            case PpoModelVariant.PPO_RESNET50_UNFROZEN:
                return self._create_resnet_model(vec_env, n_steps, batch_size, n_epochs, False, ResNetFeatureExtractor,
                                                 learning_rate, **kwargs)
            case PpoModelVariant.PPO_RESNET50_FROZEN:
                return self._create_resnet_model(vec_env, n_steps, batch_size, n_epochs, True, ResNetFeatureExtractor,
                                                 learning_rate, **kwargs)
            case _:
                raise ValueError(f"Unknown PPO model variant: {self.model_variant}")

    def _create_ppo_without_backbone(self,
                                     vec_env: VecEnv,
                                     n_steps: int,
                                     batch_size: int,
                                     n_epochs: int,
                                     learning_rate: Union[float, Schedule],
                                     **kwargs) -> PPO:
        return PPO("CnnPolicy",
                   env=vec_env,
                   n_steps=n_steps,
                   learning_rate=learning_rate,
                   batch_size=batch_size,
                   n_epochs=n_epochs,
                   policy_kwargs={"normalize_images": False},
                   verbose=1)

    def _create_resnet_model(self, vec_env: VecEnv,
                             n_steps: int,
                             batch_size: int,
                             n_epochs: int,
                             freeze_backbone: bool,
                             extractor_class: Type[Union[ResNetFeatureExtractor, ResNet18FeatureExtractor]],
                             learning_rate: Union[float, Schedule] = 3e-4,
                             feature_dim: int = 512,
                             **kwargs) -> PPO:
        return PPO("CnnPolicy",
                   env=vec_env,
                   n_steps=n_steps,
                   learning_rate=learning_rate,
                   batch_size=batch_size,
                   n_epochs=n_epochs,
                   policy_kwargs=self._create_resnet_extractor(extractor_class, feature_dim, freeze_backbone),
                   verbose=1)

    def _create_resnet_extractor(
            self,
            extractor_class: Type[Union[ResNetFeatureExtractor, ResNet18FeatureExtractor]],
            feature_dim: int,
            freeze_backbone: bool = True) -> dict[str, type[ResNetFeatureExtractor] | dict[str, int | bool]]:
        return {
            "features_extractor_class": extractor_class,
            "features_extractor_kwargs": {
                "features_dim": feature_dim,
                "pretrained": True,
                "freeze_backbone": freeze_backbone
            },
            "normalize_images": False
        }
