from typing import Union

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from training.stable_baselines.hyperparameter.runtime_hyperparams import RuntimeParams


class RuntimeParamsBuilder:
    def __init__(self, experiment_name: str, run_name: str, total_training_steps: int, num_vector_envs: int):
        self._params: RuntimeParams = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "total_training_steps": total_training_steps,
            "num_vector_envs": num_vector_envs,
            # Defaults
            "random_seed": 42,
            "vec_env_cls": DummyVecEnv,
            "use_worker_pool": False,
            "num_juror_workers": 0,
            "use_local_juror": True,
            "render_mode": "skip",
            "render_save_dir": "./renders/",
            "evaluation_seed": 67,
            "evaluation_interval": 10000,
            "evaluation_deterministic": True,
            "evaluation_visual_history": False,
            "evaluation_visual_history_max_images": 20,
            "evaluation_visual_history_max_size": 200,
            "evaluation_render_mode": "skip",
            "evaluation_render_save_dir": "./evaluation/renders/",
            "evaluation_model_save_dir": "./evaluation/models/",
            "evaluation_log_path": "./evaluation/logs/"
        }

    def with_random_seed(self, seed: int) -> "RuntimeParamsBuilder":
        self._params["random_seed"] = seed
        return self

    def with_resource_settings(self,
                               use_worker_pool: bool,
                               num_juror_workers: int,
                               vec_env_cls: Union[type[DummyVecEnv] | type[SubprocVecEnv] | str],
                               use_local_juror: bool = True) -> "RuntimeParamsBuilder":
        self._params["use_worker_pool"] = use_worker_pool
        self._params["num_juror_workers"] = num_juror_workers
        self._params["use_local_juror"] = use_local_juror
        
        if isinstance(vec_env_cls, str):
             self._params["vec_env_cls"] = SubprocVecEnv if vec_env_cls == "SubprocVecEnv" else DummyVecEnv
        else:
            self._params["vec_env_cls"] = vec_env_cls
            
        return self

    def with_rendering(self, render_mode: str, save_dir: str) -> "RuntimeParamsBuilder":
        self._params["render_mode"] = render_mode
        self._params["render_save_dir"] = save_dir
        return self

    def with_evaluation(self,
                        interval: int,
                        visual_history: bool = True,
                        visual_history_max_images: int = 20,
                        visual_history_max_size: int = 200,
                        seed: int = 67) -> "RuntimeParamsBuilder":
        self._params["evaluation_interval"] = interval
        self._params["evaluation_visual_history"] = visual_history
        self._params["evaluation_visual_history_max_images"] = visual_history_max_images
        self._params["evaluation_visual_history_max_size"] = visual_history_max_size
        self._params["evaluation_seed"] = seed
        return self

    def build(self) -> RuntimeParams:
        return self._params