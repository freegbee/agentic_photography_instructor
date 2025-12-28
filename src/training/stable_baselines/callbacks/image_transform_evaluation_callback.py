import logging
from typing import List, Optional

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

from training import mlflow_helper
from training.stable_baselines.callbacks.reporting_utils import ReportingUtils

logger = logging.getLogger(__name__)


class ImageTransformEvaluationCallback(EvalCallback):
    """
    Erweiterung des EvalCallback, um eine benutzerdefinierte Evaluationsschleife zu implementieren.
    """

    def __init__(self, stats_key: str, *args, **kwargs):
        """
        Erweiterung des EvalCallback, um eine benutzerdefinierte Evaluationsschleife zu implementieren.
        param stats_key: Schlüssel in den Infos des Environments, unter dem die Endwerte der Episode zu finden ist.
        """
        super().__init__(*args, **kwargs)
        self._stats_key = stats_key
        self.last_eval_step = 0
        self.evaluation_idx = 0


    def _get_model_checksum(self) -> Optional[float]:
        try:
            # leichter Check: Summe des ersten Parameters (nur für Debugging)
            import torch
            params = []
            if hasattr(self.model, "policy"):
                params = list(self.model.policy.parameters())
            elif hasattr(self.model, "parameters"):
                params = list(self.model.parameters())
            if params:
                return float(params[0].data.sum().cpu().item())
        except Exception:
            pass
        return None

    def _run_custom_evaluation(self) -> dict:
        """"
        Evaluationsschleife, welche das aktuelle Modell im übergebenen Evaluations-Environment ausführt.
        Es werden hier immer alle Episoden durchlaufen, bis die gewünschte Anzahl an Episoden erreicht ist.
        """
        env = self.eval_env  # Evaluations-Environment
        n_target = int(self.n_eval_episodes)  # Anzahl Episoden, die evaluiert werden sollen
        n_envs = getattr(env, "num_envs", 1)  # Anzahl paralleler Environments - falls Vektor-Env genutzt wird
        obs = env.reset()  # Environment zurücksetzen und von vorne beginnen

        # Sammeln aller Episoden-Statistiken - kumulieren erfolg später
        # NICE Lässt sich wohl auch so lösen, dass es nicht so viel Speicher braucht (laufend kumulieren, statt am Ende nochmals über alles loopen)
        collected_evaluation_episode_infos: List[dict] = []
        collected_episode = 0

        # Für jede Environment Episoden sammeln, bis n_target erreicht ist
        # n_target ist die Anzahl der zu durchlaufenden Datensätze. Beim Evaluation Callback ist das die Anzahl der Bilder des Validationssets.
        while collected_episode < n_target:
            # Prediction des Modells für die aktuelle Beobachtung
            # Es ist Plural, falls die evaluation auch mit Vektor-Env läuft
            # Trainiertes Modell predicted die Aktion(en) für die aktuelle(n) Beobachtung(en)
            actions, _ = self.model.predict(obs, deterministic=self.deterministic)
            # Führe die Aktion(en) im Environment aus - erhalte neue Beobachtung, Belohnung, ob Episode zu Ende ist und evtl. Infos.
            # Die Infos haben unseren Spezialwerte (success, initial_score und so weiter)
            obs, rewards, dones, infos = env.step(actions)

            # make infos iterable for single-env
            infos_list = infos if isinstance(infos, list) else [infos]

            # Loop über alle Environments (bei Vektor-Env) - sonst halt genau 1 mal
            # Wir haben dann Zugriff auf die ergebnisse der actions und rewards und infos etc. für jede Environment bier counter
            for i in range(n_envs):
                r = float(rewards[i]) if isinstance(rewards, (list, np.ndarray)) else float(rewards)

                done = dones[i] if isinstance(dones, (list, np.ndarray)) else dones
                if done:
                    # Wenn done die Episoden-Statistiken merken
                    episode_info = infos_list[i] if i < len(infos_list) else {}
                    collected_evaluation_episode_infos.append(dict(episode_info[self._stats_key]))
                    collected_episode += 1

                    # Abbrechen mit dem Sammeln, wenn Ziel erreicht
                    if collected_episode >= n_target:
                        break

        return ReportingUtils.create_mlflow_metrics(rollout_idx=self.evaluation_idx,
                                                    metrics_collection=collected_evaluation_episode_infos,
                                                    metric_key_prefix="eval")

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.num_timesteps - self.last_eval_step) >= self.eval_freq:
            model_checksum = self._get_model_checksum()
            logger.debug("Starting evaluation idx=%d at timestep=%d model_id=%s checksum=%s",
                         self.evaluation_idx, self.num_timesteps, id(self.model), model_checksum)
            self.last_eval_step = self.num_timesteps
            metrics = self._run_custom_evaluation()
            metrics["eval_model_checksum"] = model_checksum
            try:
                mlflow_helper.log_batch_metrics(metrics, step=self.evaluation_idx)
            except Exception:
                logger.warning("Failed to log rollout metrics")
                pass
            # Beispiel: loggen über SB3 logger (oder mlflow, je nach Projekt)
            logger.info("Eval step for model with checksum %s : %d: %s", model_checksum, self.evaluation_idx, str(metrics))

            # optional: save best model wie EvalCallback es tut
            if self.best_mean_reward is None or metrics["eval_mean_reward"] > self.best_mean_reward:
                self.best_mean_reward = metrics["eval_mean_reward"]
                if self.best_model_save_path is not None:
                    self.model.save(f"{self.best_model_save_path}/best_model")

            # Nächster Step merken
            self.evaluation_idx += 1
        return True
