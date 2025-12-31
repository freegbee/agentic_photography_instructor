import time
from stable_baselines3.common.callbacks import BaseCallback

from training import mlflow_helper

class MlflowPerformanceCallback(BaseCallback):
    """
    Ein Custom Callback für Stable Baselines3, der Performance-Metriken
    (Dauer, FPS, Step-Time) pro Rollout an MLflow sendet.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_start_time = None

    def _on_rollout_start(self) -> None:
        """
        Wird aufgerufen, bevor ein neuer Rollout (Datensammlung) beginnt.
        Hier starten wir die Stoppuhr.
        """
        self.rollout_start_time = time.time()

    def _on_rollout_end(self) -> None:
        """
        Wird aufgerufen, nachdem der Rollout beendet ist und bevor das Training (Gradient Update) beginnt.
        Hier berechnen wir die Metriken und senden sie an MLflow.
        """
        if self.rollout_start_time is None:
            return

        # Dauer des aktuellen Rollouts berechnen
        current_time = time.time()
        duration = current_time - self.rollout_start_time
        
        # Anzahl der Schritte in diesem Rollout ermitteln.
        # Bei On-Policy Algorithmen (PPO, A2C) ist dies self.model.n_steps.
        n_steps = 0
        if hasattr(self.model, "n_steps"):
            n_steps = self.model.n_steps
        
        # Metriken loggen, wenn Schritte vorhanden sind
        if n_steps > 0:
            fps = (n_steps * self.training_env.num_envs) / duration
            time_per_step = duration / (n_steps * self.training_env.num_envs)

            # Wir nutzen 'perf' als Prefix, um es in MLflow gut gruppieren zu können
            mlflow_helper.log_metric("perf/rollout_duration_s", duration, step=self.num_timesteps)
            mlflow_helper.log_metric("perf/fps_current_rollout", fps, step=self.num_timesteps)
            mlflow_helper.log_metric("perf/time_per_step_s", time_per_step, step=self.num_timesteps)

    def _on_step(self) -> bool:
        # Wir loggen nicht jeden einzelnen Step, da dies den Overhead durch HTTP-Requests zu MLflow massiv erhöhen würde.
        return True