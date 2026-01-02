from enum import Enum

from training.stable_baselines.environment.image_transform_env import ImageTransformEnv
from training.stable_baselines.environment.image_optimization_env import ImageOptimizationEnv


class WellDefinedEnvironment(Enum):
    IMAGE_DEDEGRATION = (ImageTransformEnv, "Degradation")
    IMAGE_OPTIMIZATION = (ImageOptimizationEnv, "Optimization")

    def __init__(self, env_class, human_readable_name):
        self.env_class = env_class
        self.human_readable_name = human_readable_name