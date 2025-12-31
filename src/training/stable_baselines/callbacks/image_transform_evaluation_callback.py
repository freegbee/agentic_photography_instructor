import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import mlflow
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

from training import mlflow_helper
from training.stable_baselines.callbacks.reporting_utils import ReportingUtils

logger = logging.getLogger(__name__)


class VisualSummaryLogger:
    """
    Kapselt die Logik zur Erstellung und zum Logging des visuellen Mosaiks.
    """
    def __init__(self, max_tile_size: int = 150):
        self._max_tile_size = max_tile_size

    def log_summary(self, histories: List[List[np.ndarray]], evaluation_idx: int):
        """
        Erstellt ein Mosaik aus den Bildverläufen und lädt es als MLflow Artefakt hoch.
        """
        try:
            mosaic = self._generate_mosaic(histories)
            if mosaic is None:
                return

            with tempfile.TemporaryDirectory() as tmp_dir:
                filename = f"eval_mosaic_{evaluation_idx:03d}.jpg"
                filepath = Path(tmp_dir) / filename
                # JPG-Qualität (0-100). Höher ist besser. 85 ist ein guter Standardwert.
                cv2.imwrite(str(filepath), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 85])
                mlflow.log_artifact(str(filepath), artifact_path="evaluation_images")
                logger.info(f"Logged evaluation mosaic: {filename}")

        except Exception as e:
            logger.error(f"Failed to generate or log visual summary: {e}", exc_info=True)

    def _generate_mosaic(self, histories: List[List[np.ndarray]]) -> Optional[np.ndarray]:
        rows = []
        for history in histories:
            row_images = []
            for img in history:
                if img is None:
                    continue
                
                # Resize auf Kachelgröße (falls noch nicht geschehen oder abweichend)
                h, w = img.shape[:2]
                scale = min(self._max_tile_size / h, self._max_tile_size / w)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Auf quadratischen Canvas zentrieren (für sauberes Grid)
                canvas = np.zeros((self._max_tile_size, self._max_tile_size, 3), dtype=np.uint8)
                y_offset = (self._max_tile_size - new_h) // 2
                x_offset = (self._max_tile_size - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                row_images.append(canvas)
            
            if row_images:
                rows.append(np.hstack(row_images))

        if not rows:
            return None

        # Zeilen auf gleiche Breite bringen
        max_width = max(row.shape[1] for row in rows)
        padded_rows = []
        for row in rows:
            h, w = row.shape[:2]
            if w < max_width:
                padding = np.zeros((h, max_width - w, 3), dtype=np.uint8)
                row = np.hstack([row, padding])
            padded_rows.append(row)
        
        return np.vstack(padded_rows)


class ImageTransformEvaluationCallback(EvalCallback):
    """
    Erweiterung des EvalCallback, um eine benutzerdefinierte Evaluationsschleife zu implementieren.
    """

    def __init__(self, stats_key: str, num_images_to_log: int = 5, tile_max_size: int = 150, *args, **kwargs):
        """
        Erweiterung des EvalCallback, um eine benutzerdefinierte Evaluationsschleife zu implementieren.
        param stats_key: Schlüssel in den Infos des Environments, unter dem die Endwerte der Episode zu finden ist.
        param num_images_to_log: Anzahl der Episoden, für die ein visueller Verlauf (Mosaik) erstellt werden soll.
        """
        super().__init__(*args, **kwargs)
        self._stats_key = stats_key
        self.num_images_to_log = num_images_to_log
        self.last_eval_step = 0
        self.evaluation_idx = 0
        self._visual_logger = VisualSummaryLogger(max_tile_size=tile_max_size)


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

        # History-Aufzeichnung aktivieren (für die ersten N Bilder), falls gewünscht
        self._set_history_recording(env, self.num_images_to_log > 0)

        # Sampler explizit auf Anfang setzen, um Doppel-Reset-Probleme zu vermeiden
        self._reset_sampler_to_start(env)

        obs = env.reset()  # Environment zurücksetzen und von vorne beginnen

        # Sammeln aller Episoden-Statistiken - kumulieren erfolg später
        # NICE Lässt sich wohl auch so lösen, dass es nicht so viel Speicher braucht (laufend kumulieren, statt am Ende nochmals über alles loopen)
        collected_evaluation_episode_infos: List[dict] = []
        collected_image_histories: List[List[np.ndarray]] = []
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
                    
                    # Bild-Historie sammeln, falls vorhanden und Limit noch nicht erreicht
                    if "image_history" in episode_info and len(collected_image_histories) < self.num_images_to_log:
                        collected_image_histories.append(episode_info["image_history"])
                        
                        # Wenn Limit erreicht, Aufzeichnung deaktivieren um Ressourcen zu sparen
                        if len(collected_image_histories) >= self.num_images_to_log:
                            self._set_history_recording(env, False)

                    # Abbrechen mit dem Sammeln, wenn Ziel erreicht
                    if collected_episode >= n_target:
                        break
        
        # Mosaik generieren und loggen
        if collected_image_histories:
            self._visual_logger.log_summary(collected_image_histories, self.evaluation_idx)

        return ReportingUtils.create_mlflow_metrics(rollout_idx=self.evaluation_idx,
                                                    metrics_collection=collected_evaluation_episode_infos,
                                                    metric_key_prefix="eval")

    def _set_history_recording(self, env, enable: bool):
        try:
            if hasattr(env, "env_method"):
                env.env_method("set_keep_image_history", enable)
        except AttributeError:
            pass  # Ignore if wrapper is not present or method does not exist

    def _reset_sampler_to_start(self, env):
        """Versucht, den Sampler im Environment auf den Anfang zurückzusetzen."""
        try:
            if hasattr(env, "env_method"):
                # Erwartet, dass ImageTransformEnv eine Methode 'reset_sampler()' hat
                env.env_method("reset_sampler")
        except Exception:
            # Falls die Methode nicht existiert, machen wir einfach weiter (Fallback auf Standard-Verhalten)
            pass

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
