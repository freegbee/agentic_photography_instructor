class RunningMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cumulated_episodes: int = 0
        self.cumulated_episode_len: int = 0
        self.cumulated_reward: float = 0
        self.cumulated_successes: int = 0
        self.cumulated_score: float = 0
        self.cumulated_initial_score: float = 0
        self.cumulated_matches_expected: int = 0

    def update(self, episode: dict):
        self.cumulated_episodes += 1
        self.cumulated_episode_len += int(episode.get("l", 0.0))
        self.cumulated_reward += float(episode.get("r", 0.0))
        if episode.get("success", False):
            self.cumulated_successes += 1
        if episode.get("matches_expected", False):
            self.cumulated_matches_expected += 1
        if episode.get("score") is not None:
            self.cumulated_score += float(episode.get("score"))
        if episode.get("initial_score") is not None:
            self.cumulated_initial_score += float(episode.get("initial_score"))

    def create_mlflow_metrics(self, rollout_idx, metric_key_prefix) -> dict[str, int | float]:
        cumulated_episodes = self.cumulated_episodes

        mean_episode_len = self.cumulated_episode_len / cumulated_episodes if cumulated_episodes > 0 else 0.0
        mean_reward = self.cumulated_reward / cumulated_episodes if cumulated_episodes > 0 else 0.0
        success_rate = self.cumulated_successes / cumulated_episodes if cumulated_episodes > 0 else 0.0
        mean_score = self.cumulated_score / cumulated_episodes if cumulated_episodes > 0 else 0.0
        mean_initial_score = self.cumulated_initial_score / cumulated_episodes if cumulated_episodes > 0 else 0.0

        metrics = {
            f"{metric_key_prefix}/rollout_index": rollout_idx,
            f"{metric_key_prefix}/episodes": cumulated_episodes,
            f"{metric_key_prefix}/mean_episodes_len": mean_episode_len,
            f"{metric_key_prefix}/mean_reward": mean_reward,
            f"{metric_key_prefix}/images_with_success": int(self.cumulated_successes),
            f"{metric_key_prefix}/success_rate": float(success_rate),
            f"{metric_key_prefix}/matches_expected": float(self.cumulated_matches_expected),
            f"{metric_key_prefix}/mean_score": float(mean_score),
            f"{metric_key_prefix}/mean_initial_score": float(mean_initial_score)
        }
        return metrics


class ReportingUtils:

    @staticmethod
    def create_mlflow_metrics(rollout_idx, metrics_collection, metric_key_prefix) -> dict[str, int | float]:
        metrics = RunningMetrics()
        for episode in metrics_collection:
            metrics.update(episode)
        return metrics.create_mlflow_metrics(rollout_idx, metric_key_prefix)
