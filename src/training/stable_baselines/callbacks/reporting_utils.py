class ReportingUtils:

    @staticmethod
    def create_mlflow_metrics(rollout_idx, metrics_collection, metric_key_prefix) -> dict[str, int | float]:
        cumulated_episodes: int = 0
        cumulated_episode_len: int = 0
        cumulated_reward: float = 0
        cumulated_successes: int = 0
        cumulated_score: float = 0
        cumulated_initial_score: float = 0
        cumulated_matches_expected: int = 0

        for episode in metrics_collection:
            cumulated_episodes += 1
            cumulated_episode_len += int(episode.get("l", 0.0))
            cumulated_reward += float(episode.get("r", 0.0))
            if episode.get("success", False):
                cumulated_successes += 1
            if episode.get("matches_expected", False):
                cumulated_matches_expected += 1
            if episode.get("score") is not None:
                cumulated_score += float(episode.get("score"))
            if episode.get("initial_score") is not None:
                cumulated_initial_score += float(episode.get("initial_score"))

        mean_episode_len = cumulated_episode_len / cumulated_episodes if cumulated_episodes > 0 else 0.0
        mean_reward = cumulated_reward / cumulated_episodes if cumulated_episodes > 0 else 0.0
        success_rate = cumulated_successes / cumulated_episodes if cumulated_episodes > 0 else 0.0
        mean_score = cumulated_score / cumulated_episodes if cumulated_episodes > 0 else 0.0
        mean_initial_score = cumulated_initial_score / cumulated_episodes if cumulated_episodes > 0 else 0.0

        metrics = {
            f"{metric_key_prefix}_rollout_index": rollout_idx,
            f"{metric_key_prefix}_episodes": cumulated_episodes,
            f"{metric_key_prefix}_mean_episodes_len": mean_episode_len,
            f"{metric_key_prefix}_mean_reward": mean_reward,
            f"{metric_key_prefix}_images_with_success": int(cumulated_successes),
            f"{metric_key_prefix}_success_rate": float(success_rate),
            f"{metric_key_prefix}_matches_expected": float(cumulated_matches_expected),
            f"{metric_key_prefix}_mean_score": float(mean_score),
            f"{metric_key_prefix}_mean_initial_score": float(mean_initial_score)
        }
        return metrics