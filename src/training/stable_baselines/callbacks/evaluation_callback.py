import logging

from stable_baselines3.common.callbacks import EvalCallback

logger = logging.getLogger(__name__)

class EvaluationCallback(EvalCallback):

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            logger.info(f"EvaluationCallback: Evaluation triggered at step {self.num_timesteps}")

        cont_training = super()._on_step()
        if not cont_training:
            # Training wird abgebrochen
            logger.info(f"EvaluationCallback: Training wird abgebrochen bei step {self.num_timesteps}")
        return cont_training

    def _on_rollout_end(self) -> None:
        logger.info(f"EvaluationCallback: Rollout ended at step {self.num_timesteps}")
        return super()._on_rollout_end()

