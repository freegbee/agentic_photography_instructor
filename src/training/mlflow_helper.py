import logging
import threading
from typing import Optional

import mlflow
from mlflow.entities import Run, Experiment

from training.mlflow_utils import MLflowTagsBuilder

logger = logging.getLogger(__name__)

class MlflowHelper:
    _instance: Optional["MlflowHelper"] = None
    _lock = threading.Lock()

    def __new__(cls, active_run: Optional[Run] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, active_run: Optional[Run] = None):
        logger.info(f"++++ MlflowHelper.__init__ called")
        self.mlflow = mlflow
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.active_run: Optional[Run] = active_run
        self.local_logging = False

    def with_active_run(self, active_run: Run) -> 'MlflowHelper':
        if not active_run:
            raise ValueError("active_run must not be None")
        if self.active_run is not None:
            raise ValueError("active_run is already set")
        self.active_run = active_run
        return self

    def without_local_logging(self) -> 'MlflowHelper':
        self.local_logging = False
        return self

    def with_local_logging(self) -> 'MlflowHelper':
        self.local_logging = True
        return self

    def start_run(self, experiment: Experiment, run_name: str, tags_builder:MLflowTagsBuilder) -> Run:
        if self.local_logging:
            logger.info("start_run: experiment_id=%s, run_name=%s, tags=%s", experiment.experiment_id, run_name, tags_builder.build())
        run = self.mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id, tags=tags_builder.build())
        self.with_active_run(run)
        return run

    def end_run(self):
        if self.local_logging:
            logger.info("end_run called for run_id=%s", self.active_run.info.run_id if self.active_run else "None")
        if self.active_run is not None:
            self.mlflow.end_run()

    def log_metric(self, key: str, value, step: Optional[int] = None):
        if self.local_logging:
            logger.info("log_metric: key=%s, value=%s, step=%s", key, value, step)
        if not value:
            return

        if step is None:
            self.mlflow.log_metric(key=key, value=float(value))
        else:
            self.mlflow.log_metric(key=key, value=float(value), step=step)

    def log_batch_metrics(self, metrics: dict[str, float], step: Optional[int] = None):
        if self.local_logging:
            logger.info("log_batch_metrics: metrics=%s, step=%s", metrics, step)
        if step is None:
            self.mlflow.log_metrics(metrics)
        else:
            self.mlflow.log_metrics(metrics, step=step)

    def log_param(self, key: str, value):
        if self.local_logging:
            logger.info("log_param: key=%s, value=%s", key, value)
        self.mlflow.log_param(key, value)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None, step: Optional[int] = None):
        if self.local_logging:
            logger.info("log_artifact: local_path=%s, artifact_path=%s, step=%s", local_path, artifact_path, step)
        try:
            if step:
                self.mlflow.log_artifact(local_path, artifact_path=f"step/{step}/{artifact_path if artifact_path else ''}")
            else:
                self.mlflow.log_artifact(local_path, artifact_path)
            return
        except Exception as e:
            logger.warning("mlflow.log_artifact failed: %s", e)

    def log_dataset(self, dataset_id: str, annotations_file: str, images_source_path: str, context: str = "training"):
        if self.local_logging:
            logger.info("log_dataset: dataset_id=%s, source_path=%s", dataset_id, images_source_path)
        try:
            import pandas as pd
            from mlflow.data.sources import LocalArtifactDatasetSource

            ds_meta = pd.DataFrame([{
                "dataset_id": dataset_id,
                "annotations_file": annotations_file,
                "source_path": images_source_path,
            }])
            # Explicitly create source to avoid ambiguity warning
            # WICHTIG: Dies lädt KEINE Daten hoch, sondern speichert nur den Pfad als Referenz.
            source = LocalArtifactDatasetSource(images_source_path)
            dataset = self.mlflow.data.from_pandas(ds_meta, source=source, name=dataset_id)
            self.mlflow.log_input(dataset, context=context)
        except Exception as e:
            logger.warning("Failed to log dataset to MLflow: %s", e)