import logging
from stable_baselines3.common.callbacks import BaseCallback
from training import mlflow_helper
from training.stable_baselines.utils.visual_loggers import VisualStatisticsLogger

logger = logging.getLogger(__name__)


class TransformerUsageCallback(BaseCallback):
    """
    Callback zum Loggen der Transformer-Nutzung.
    - Zählt direkt die Nutzung während der Trainings-Rollouts.
    - Liest optional einen Wrapper aus, um die Nutzung während der Evaluation zu loggen.
    """
    def __init__(self, eval_env_wrapper=None, train_env_wrapper=None, verbose=0):
        super().__init__(verbose)
        self.eval_env_wrapper = eval_env_wrapper
        self.train_env_wrapper = train_env_wrapper
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
                logger.info(f"TransformerUsageCallback: Logged evaluation usage: {usage}")
                
                # Visualisierung erstellen
                self.visual_logger.log_transformer_distribution(usage, self.num_timesteps, prefix="eval")
            
            # Crop Stats logging
            if hasattr(self.eval_env_wrapper, "pop_crop_stats"):
                crop_stats = self.eval_env_wrapper.pop_crop_stats()
                if crop_stats and crop_stats['attempts'] > 0:
                    attempts = crop_stats['attempts']
                    changed = crop_stats['changed']
                    mlflow_helper.log_metric("eval_crop/attempts", attempts, step=self.num_timesteps)
                    mlflow_helper.log_metric("eval_crop/changed_ratio", changed / attempts, step=self.num_timesteps)
                    if changed > 0:
                        mlflow_helper.log_metric("eval_crop/avg_score_delta_on_change", crop_stats['score_delta_sum'] / changed, step=self.num_timesteps)

            # MDP Stats logging
            if hasattr(self.eval_env_wrapper, "pop_mdp_stats"):
                mdp_stats = self.eval_env_wrapper.pop_mdp_stats()
                if mdp_stats["total"] > 0:
                    mlflow_helper.log_metric("eval_mdp/count", mdp_stats["count"], step=self.num_timesteps)
                    mlflow_helper.log_metric("eval_mdp/ratio", mdp_stats["count"] / mdp_stats["total"], step=self.num_timesteps)

            elif self.eval_env_wrapper.has_data: # Sollte nicht passieren wenn usage None ist, aber zur Sicherheit
                logger.warning("TransformerUsageCallback: Eval wrapper has data flag but returned empty usage.")
        
        return True

    def _on_rollout_end(self) -> None:
        # Training Usage direkt vom Environment-Wrapper abholen
        # Wir nutzen bevorzugt den explizit übergebenen Wrapper, um SB3-interne Wrapper (z.B. VecTransposeImage) zu umgehen.
        env = self.train_env_wrapper if self.train_env_wrapper else self.model.get_env()
        
        # Prüfen, ob die Methode existiert (falls SB3 intern weitere Wrapper hinzufügt, werden Attribute meist durchgereicht)
        if hasattr(env, "pop_usage"):
            usage = env.pop_usage()
            if usage:
                metrics = {f"train_transformer_usage/{label}": count for label, count in usage.items()}
                metrics["train_transformer_usage/total"] = sum(usage.values())
                mlflow_helper.log_batch_metrics(metrics, step=self.num_timesteps)
                logger.info(f"TransformerUsageCallback: Logged training usage: {usage}")
                
                # Visualisierung erstellen
                self.visual_logger.log_transformer_distribution(usage, self.num_timesteps, prefix="train")
            
            # Crop Stats logging
            if hasattr(env, "pop_crop_stats"):
                crop_stats = env.pop_crop_stats()
                if crop_stats and crop_stats['attempts'] > 0:
                    attempts = crop_stats['attempts']
                    changed = crop_stats['changed']
                    mlflow_helper.log_metric("train_crop/attempts", attempts, step=self.num_timesteps)
                    mlflow_helper.log_metric("train_crop/changed_ratio", changed / attempts, step=self.num_timesteps)
                    if changed > 0:
                        mlflow_helper.log_metric("train_crop/avg_score_delta_on_change", crop_stats['score_delta_sum'] / changed, step=self.num_timesteps)

            # MDP Stats logging
            if hasattr(env, "pop_mdp_stats"):
                mdp_stats = env.pop_mdp_stats()
                if mdp_stats["total"] > 0:
                    mlflow_helper.log_metric("train_mdp/count", mdp_stats["count"], step=self.num_timesteps)
                    mlflow_helper.log_metric("train_mdp/ratio", mdp_stats["count"] / mdp_stats["total"], step=self.num_timesteps)

            else:
                logger.warning("TransformerUsageCallback: No usage data collected during rollout (usage is empty).")
        else:
            logger.warning("TransformerUsageCallback: Environment does not support pop_usage.")