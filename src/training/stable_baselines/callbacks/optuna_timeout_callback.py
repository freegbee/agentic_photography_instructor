import logging
import time
import optuna
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class OptunaTimeoutCallback(BaseCallback):
    """
    Callback, der das Training abbricht, wenn ein definiertes Zeitlimit überschritten wird.
    Dies verhindert, dass einzelne Trials die gesamte Optimierung blockieren.
    """

    def __init__(self, timeout_seconds: int, verbose: int = 0):
        super().__init__(verbose)
        self.timeout_seconds = timeout_seconds
        self.start_time = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Prüfen, ob das Zeitlimit überschritten wurde
        if (time.time() - self.start_time) > self.timeout_seconds:
            if self.verbose > 0:
                logger.warning(f"TimeoutCallback: Stopping training after {self.timeout_seconds} seconds.")
            raise optuna.exceptions.TrialPruned(f"Timeout reached: {self.timeout_seconds} seconds")
        return True