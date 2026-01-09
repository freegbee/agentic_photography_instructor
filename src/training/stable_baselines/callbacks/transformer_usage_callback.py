from stable_baselines3.common.callbacks import BaseCallback
from training import mlflow_helper
from training.stable_baselines.utils.visual_loggers import VisualStatisticsLogger


class TransformerUsageCallback(BaseCallback):
    """
    Callback zum Loggen der Transformer-Nutzung.
    - Zählt direkt die Nutzung während der Trainings-Rollouts.
    - Liest optional einen Wrapper aus, um die Nutzung während der Evaluation zu loggen.
    """
    def __init__(self, eval_env_wrapper=None, verbose=0):
        super().__init__(verbose)
        self.eval_env_wrapper = eval_env_wrapper
        self.visual_logger = VisualStatisticsLogger()

    def _on_rollout_start(self) -> None:
        # Nicht mehr nötig, da der Wrapper (TransformerUsageVecEnvWrapper) den Buffer bei pop_usage() leert.
        pass

    def _on_step(self) -> bool:
        # Evaluation Usage prüfen (falls Wrapper vorhanden und Daten bereitliegen)
        # Da der EvalCallback innerhalb eines Steps läuft, können wir hier prüfen, ob er fertig ist.
        if self.eval_env_wrapper:
            usage = self.eval_env_wrapper.pop_usage()
            if usage:
                metrics = {f"eval_transformer_usage/{label}": count for label, count in usage.items()}
                metrics["eval_transformer_usage/total"] = sum(usage.values())
                mlflow_helper.log_batch_metrics(metrics, step=self.num_timesteps)
                
                # Visualisierung erstellen
                self.visual_logger.log_transformer_distribution(usage, self.num_timesteps, prefix="eval")
        
        return True

    def _on_rollout_end(self) -> None:
        # Training Usage direkt vom Environment-Wrapper abholen
        # self.model.get_env() liefert das gewrappte Environment zurück
        env = self.model.get_env()
        
        # Prüfen, ob die Methode existiert (falls SB3 intern weitere Wrapper hinzufügt, werden Attribute meist durchgereicht)
        if hasattr(env, "pop_usage"):
            usage = env.pop_usage()
            if usage:
                metrics = {f"train_transformer_usage/{label}": count for label, count in usage.items()}
                metrics["train_transformer_usage/total"] = sum(usage.values())
                mlflow_helper.log_batch_metrics(metrics, step=self.num_timesteps)
                
                # Visualisierung erstellen
                self.visual_logger.log_transformer_distribution(usage, self.num_timesteps, prefix="train")