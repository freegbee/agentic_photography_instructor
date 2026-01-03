from training.stable_baselines.environment.welldefined_environments import WellDefinedEnvironment
from training.stable_baselines.hyperparameter.task_hyperparams import TaskParams


class TaskParamsBuilder:
    def __init__(self, core_env: WellDefinedEnvironment, transformer_labels: list[str], max_transformations: int):
        self._params: TaskParams = {
            "core_env": core_env,
            "transformer_labels": transformer_labels,
            "max_transformations": max_transformations,
            # Defaults
            "success_bonus": 1.0,
            "use_multi_step_wrapper": False,
            "steps_per_episode": 1,
            "multi_step_intermediate_reward": False,
            "multi_step_reward_shaping": False
        }

    def with_rewards(self, success_bonus: float) -> "TaskParamsBuilder":
        self._params["success_bonus"] = success_bonus
        return self

    def with_multi_step_logic(self,
                              steps_per_episode: int,
                              intermediate_reward: bool = False,
                              reward_shaping: bool = False) -> "TaskParamsBuilder":
        self._params["use_multi_step_wrapper"] = True
        self._params["steps_per_episode"] = steps_per_episode
        self._params["multi_step_intermediate_reward"] = intermediate_reward
        self._params["multi_step_reward_shaping"] = reward_shaping
        return self

    def build(self) -> TaskParams:
        return self._params