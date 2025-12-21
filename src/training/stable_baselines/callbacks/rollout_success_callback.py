import logging
from typing import List

from stable_baselines3.common.callbacks import BaseCallback

from training import mlflow_helper

logger = logging.getLogger(__name__)


class RolloutSuccessCallback(BaseCallback):
    """
    Loggt nach jedem Rollout:
      - Anzahl neuer abgeschlossener Episoden
      - Anzahl erfolgreicher Episoden
      - Erfolgsrate
      - Summe und Mittel der kumulierten Rewards (\"r\" / \"reward\")
    Unterstützt verschiedene Strukturen in ep_info_buffer (direkt, unter 'episode' oder unter dem übergebenen stats_key).
    """

    def __init__(self, stats_key: str, episode_stats_key: str, verbose: int = 0):
        super().__init__(verbose)
        self._stats_key = stats_key
        self.episode_stats_key = episode_stats_key
        self._last_ep_info_len = 0
        self.rollout_idx: int = 0
        self._collected_episodes: List[dict] = []

    def _on_step(self) -> bool:
        infos = None
        if "infos" in self.locals:
            infos = self.locals["infos"]
        elif "info" in self.locals:
            infos = [self.locals["info"]]

        if infos is None:
            return True

        # Normalisiere auf eine Liste von dicts
        if isinstance(infos, dict):
            infos = [infos]

        try:
            for info in infos:
                if not isinstance(info, dict):
                    continue
                # Nur dann sammeln, wenn vermutlich Episodeninfos vorliegen
                if self.episode_stats_key in info:
                    # Kopiere, um Seiteneffekte zu vermeiden
                    logger.debug("Sammle Episode info: %s", info)
                    self._collected_episodes.append(dict(info[self.episode_stats_key]))
        except Exception:
            logger.warning("Failed to read infos in _on_step", exc_info=True)

        return True

    def _on_rollout_end(self) -> None:
        cumulated_episodes: int = 0
        cumulated_episode_len: int = 0
        cumulated_reward: float = 0
        cumulated_successes: int = 0
        cumulated_score: float = 0
        cumulated_initial_score: float = 0

        mean_reward = 0
        for episode in self._collected_episodes:
            cumulated_episodes += 1
            cumulated_episode_len += int(episode.get("l", 0.0))
            cumulated_reward += float(episode.get("r", 0.0))
            if episode.get("success", False):
                cumulated_successes += 1
            if episode.get("score") is not None:
                cumulated_score += float(episode.get("score"))
            if episode.get("initial_score") is not None:
                cumulated_initial_score += float(episode.get("initial_score"))
        self._collected_episodes.clear()

        mean_episode_len = cumulated_episode_len / cumulated_episodes if cumulated_episodes > 0 else 0.0
        mean_reward = cumulated_reward / cumulated_episodes if cumulated_episodes > 0 else 0.0
        success_rate = cumulated_successes / cumulated_episodes if cumulated_episodes > 0 else 0.0
        mean_score = cumulated_score / cumulated_episodes if cumulated_episodes > 0 else 0.0
        mean_initial_score = cumulated_initial_score / cumulated_episodes if cumulated_episodes > 0 else 0.0

        metrics = {
            "rollout_index": self.rollout_idx,
            "rollout_episodes": cumulated_episodes,
            "rollout_mean_episodes_len": mean_episode_len,
            "rollout_mean_reward": mean_reward,
            "rollout_images_with_success": int(cumulated_successes),
            "rollout_success_rate": float(success_rate),
            "rollout_mean_score": float(mean_score),
            "rollout_mean_initial_score": float(mean_initial_score)
        }
        try:
            mlflow_helper.log_batch_metrics(metrics, step=self.rollout_idx)
            logger.info("Finished rollout %s" % str(metrics))


        except Exception:
            logger.warning("Failed to log rollout metrics")
            pass

        self.rollout_idx += 1
