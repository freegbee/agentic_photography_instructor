from typing import TypedDict

from training.stable_baselines.environment.welldefined_environments import WellDefinedEnvironment


class TrainingParams(TypedDict, total=False):
    experiment_name: str
    run_name: str
    use_local_juror: bool
    random_seed: int
    core_env: WellDefinedEnvironment
    num_vector_envs: int
    max_transformations: int
    total_training_steps: int
    render_mode: str
    render_save_dir: str
    evaluation_seed: int
    evaluation_interval: int
    evaluation_deterministic: bool
    evaluation_visual_history: bool
    evaluation_visual_history_max_images: int
    evaluation_visual_history_max_size: int
    evaluation_render_mode: str
    evaluation_render_save_dir: str
    evaluation_model_save_dir: str
    evaluation_log_path: str
    # Multi-step wrapper parameters
    use_multi_step_wrapper: bool  # Whether to use the multi-step wrapper
    steps_per_episode: int  # Number of transformations per episode (default: 2)
    multi_step_intermediate_reward: bool  # Give small rewards at intermediate steps (default: False)
    multi_step_reward_shaping: bool  # Use reward shaping for intermediate steps (default: False)
