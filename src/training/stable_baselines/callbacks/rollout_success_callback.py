import logging
from typing import List

from stable_baselines3.common.callbacks import BaseCallback

from training import mlflow_helper
from training.stable_baselines.callbacks.reporting_utils import ReportingUtils

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

    def __init__(self, training_episode_stats_key: str, evaluation_episode_stats_key: str, verbose: int = 0):
        super().__init__(verbose)
        self._training_episode_stats_key = training_episode_stats_key
        self._evaluation_episode_stats_key = evaluation_episode_stats_key
        self._last_ep_info_len = 0
        self.rollout_idx: int = 0
        self._collected_training_episodes: List[dict] = []
        self._collected_evaluation_episodes: List[dict] = []

    def _on_step(self) -> bool:
        cont_training = super()._on_step()

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
                if self._training_episode_stats_key in info:
                    # Kopiere, um Seiteneffekte zu vermeiden
                    logger.debug("Sammle Episode info: %s", info)
                    self._collected_training_episodes.append(dict(info[self._training_episode_stats_key]))
                if self._evaluation_episode_stats_key in info:
                    logger.info("Sammle Evaluation Episode info: %s", info)
                    self._collected_evaluation_episodes.append(dict(info[self._evaluation_episode_stats_key]))

        except Exception:
            logger.warning("Failed to read infos in _on_step", exc_info=True)

        return cont_training

    def _on_rollout_end(self) -> None:
        if len(self._collected_training_episodes) > 0:
            self._process_collected_episodes(self._collected_training_episodes, "train")
        if len(self._collected_evaluation_episodes) > 0:
            self._process_collected_episodes(self._collected_evaluation_episodes, "eval")

        self.rollout_idx += 1

    def _process_collected_episodes(self, metrics_collection: List[dict], metric_key_prefix) -> None:
        metrics = ReportingUtils.create_mlflow_metrics(self.rollout_idx, metrics_collection, metric_key_prefix)
        metrics_collection.clear()
        try:
            mlflow_helper.log_batch_metrics(metrics, step=self.rollout_idx)
            logger.info("Finished %s rollout %s",  metric_key_prefix, str(metrics))
        except Exception:
            logger.warning("Failed to log rollout metrics")
            pass
