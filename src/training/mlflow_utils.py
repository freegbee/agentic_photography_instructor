import logging
import os
import re
import time
from functools import wraps
from typing import Callable, Any, Optional, List

from training.hyperparameter_registry import HyperparameterRegistry


class MLflowUtils:
    @staticmethod
    def get_tracking_client(tracking_uri: str = None):
        import mlflow
        from mlflow.tracking import MlflowClient

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        return MlflowClient()

    @staticmethod
    def log_params(mlflow_client, run_id, param_builder: 'MLflowParamBuilder'):
        for key, value in param_builder.build().items():
            mlflow_client.log_param(run_id, key, value)

    @staticmethod
    def log_tags(mlflow_client, run_id, tags_builder: 'MLflowTagsBuilder'):
        for key, value in tags_builder.build().items():
            mlflow_client.set_tag(run_id, key, value)

class MLflowParamBuilder:
    def __init__(self):
        self.params = {}
        self.hyperparams_classes: List[Any] = []

    def add_param(self, key, value):
        self.params[key] = value
        return self

    def add_param_class(self, param_class):
        self.hyperparams_classes.append(param_class)

    def with_param_class(self, param_class):
        self.hyperparams_classes.append(param_class)
        return self

    def build(self):
        camelcase_to_snakecase_pattern = re.compile(r'(?<!^)(?=[A-Z])')
        for param_class in self.hyperparams_classes:
            param_class_name = camelcase_to_snakecase_pattern.sub('_', param_class.__name__).lower()
            store = HyperparameterRegistry.get_store(param_class)
            for key, value in store.as_dict().items():
                self.params[f"{param_class_name}_{key}"] = value
        return self.params

class MLflowTagsBuilder:
    def __init__(self):
        self.tags = {}

    def add_tag(self, key, value):
        self.tags[key] = value
        return self

    def build(self):
        return self.tags


def mlflow_logging(metric_key: str = "duration_seconds", log_level: int = logging.INFO, step_arg: Optional[str] = None) -> Callable:
    """
    Decorator for instance methods (e.g. methods of AbstractTrainer).
    - Logs start/end messages with `logging`.
    - Measures duration and tries to log it to MLflow.
    - If the wrapped object has a `log_metric(key, value, step=None)` method, it will be used.
    - Otherwise falls back to `mlflow.log_metric` if there is an active run.
    - `step_arg` (optional): name of kw/attribute to use as step (e.g. 'epoch' or 'step').
    """
    def decorator(func: Callable) -> Callable:
        from training import mlflow_helper

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger(func.__module__)
            name = f"{func.__qualname__}"
            logger.log(log_level, "Start %s", name)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                logger.log(log_level, "End %s (%.4f s)", name, duration)

                # try to determine 'self' for instance methods
                self_obj = args[0] if args else None

                # try to resolve step value from kwargs or attribute on self
                step = None
                if step_arg:
                    if step_arg in kwargs:
                        step = kwargs[step_arg]
                    elif hasattr(self_obj, step_arg):
                        try:
                            step = getattr(self_obj, step_arg)
                        except Exception:
                            step = None

                try:
                    mlflow_helper.log_metric(key=metric_key, value=duration, step=step)
                except Exception as e:
                    logger.warning("Failed to log duration %s (%.4f s) with mlflow_helper singleton: %s", name, duration, e)

        return wrapper
    return decorator