import optuna
from stable_baselines3.common.callbacks import BaseCallback


class OptunaPruningCallback(BaseCallback):
    """
    Custom Callback for Optuna Pruning with StableBaselineTrainer.
    It connects to the internal EvalCallback of the trainer to report metrics.
    """
    def __init__(self, trial: optuna.Trial, verbose=0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_callback = None
        self.last_eval_count = 0

    def set_eval_callback(self, eval_callback):
        self.eval_callback = eval_callback

    def _on_step(self) -> bool:
        if self.eval_callback is not None:
            # Check if a new evaluation has occurred (EvalCallback updates n_evaluations)
            if hasattr(self.eval_callback, "n_evaluations") and self.eval_callback.n_evaluations > self.last_eval_count:
                self.last_eval_count = self.eval_callback.n_evaluations
                
                # Get the last mean reward
                current_reward = -float('inf')
                if hasattr(self.eval_callback, "last_mean_reward"):
                    current_reward = self.eval_callback.last_mean_reward
                
                # Report to Optuna
                self.trial.report(current_reward, step=self.last_eval_count)
                
                # Prune if necessary
                if self.trial.should_prune():
                    message = f"Trial {self.trial.number} pruned at eval {self.last_eval_count} with reward {current_reward}"
                    raise optuna.exceptions.TrialPruned(message)
        return True