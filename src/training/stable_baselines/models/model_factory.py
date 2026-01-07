from abc import ABC, abstractmethod
from typing import Union, Type, TypeVar, Generic, List, Dict, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import VecEnv

from training.stable_baselines.models.base_feature_extractor import ResNetFeatureExtractor, ResNet18FeatureExtractor
from training.stable_baselines.hyperparameter.ppo_model_hyperparams import PpoModelParams
from training.stable_baselines.models.model_variants import PpoModelVariant

TParams = TypeVar("TParams")


class AbstractModelFactory(ABC, Generic[TParams]):
    @abstractmethod
    def create_model(self, vec_env: VecEnv, params: TParams) -> BaseAlgorithm:
        pass


class PpoModelFactory(AbstractModelFactory):
    def create_model(self,
                     vec_env: VecEnv,
                     params: PpoModelParams) -> PPO:
        n_steps = params["n_steps"]
        batch_size = params["batch_size"]
        n_epochs = params["n_epochs"]
        learning_rate = params["model_learning_schedule"]
        net_arch = params.get("net_arch", None)
        
        # Optional parameters with SB3 defaults
        kwargs = {
            "ent_coef": params.get("ent_coef", 0.0),
            "clip_range": params.get("clip_range", 0.2),
            "gamma": params.get("gamma", 0.99),
            "gae_lambda": params.get("gae_lambda", 0.95),
            "max_grad_norm": params.get("max_grad_norm", 0.5),
        }

        match params["ppo_model_variant"]:
            case PpoModelVariant.PPO_WITHOUT_BACKBONE:
                return self._create_ppo_without_backbone(vec_env, n_steps, batch_size, n_epochs, learning_rate, net_arch=net_arch, **kwargs)
            case PpoModelVariant.PPO_RESNET18_UNFROZEN:
                return self._create_resnet_model(vec_env, n_steps, batch_size, n_epochs, False,
                                                 ResNet18FeatureExtractor, learning_rate, net_arch=net_arch, **kwargs)
            case PpoModelVariant.PPO_RESNET18_FROZEN:
                return self._create_resnet_model(vec_env, n_steps, batch_size, n_epochs, True, ResNet18FeatureExtractor,
                                                 learning_rate, net_arch=net_arch, **kwargs)
            case PpoModelVariant.PPO_RESNET50_UNFROZEN:
                return self._create_resnet_model(vec_env, n_steps, batch_size, n_epochs, False, ResNetFeatureExtractor,
                                                 learning_rate, net_arch=net_arch, **kwargs)
            case PpoModelVariant.PPO_RESNET50_FROZEN:
                return self._create_resnet_model(vec_env, n_steps, batch_size, n_epochs, True, ResNetFeatureExtractor,
                                                 learning_rate, net_arch=net_arch, **kwargs)
            case _:
                raise ValueError(f"Unknown PPO model variant: {params['ppo_model_variant']}")

    def _create_ppo_without_backbone(self,
                                     vec_env: VecEnv,
                                     n_steps: int,
                                     batch_size: int,
                                     n_epochs: int,
                                     learning_rate: Union[float, Schedule],
                                     net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                                     **kwargs
                                     ) -> PPO:
        policy_kwargs = {"normalize_images": False}
        if net_arch is not None:
            policy_kwargs["net_arch"] = net_arch
        return PPO("CnnPolicy",
                   env=vec_env,
                   n_steps=n_steps,
                   learning_rate=learning_rate,
                   batch_size=batch_size,
                   n_epochs=n_epochs,
                   policy_kwargs=policy_kwargs,
                   **kwargs,
                   verbose=1)

    def _create_resnet_model(self, vec_env: VecEnv,
                             n_steps: int,
                             batch_size: int,
                             n_epochs: int,
                             freeze_backbone: bool,
                             extractor_class: Type[Union[ResNetFeatureExtractor, ResNet18FeatureExtractor]],
                             learning_rate: Union[float, Schedule] = 3e-4,
                             feature_dim: int = 512,
                             net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                             **kwargs
                             ) -> PPO:
        policy_kwargs = self._create_resnet_extractor(extractor_class, feature_dim, freeze_backbone)
        if net_arch is not None:
            policy_kwargs["net_arch"] = net_arch

        return PPO("CnnPolicy",
                   env=vec_env,
                   n_steps=n_steps,
                   learning_rate=learning_rate,
                   batch_size=batch_size,
                   n_epochs=n_epochs,
                   policy_kwargs=policy_kwargs,
                   **kwargs,
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
