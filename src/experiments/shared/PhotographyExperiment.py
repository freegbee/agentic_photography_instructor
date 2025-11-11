import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

from mlflow import ActiveRun

from image_acquisition.acquisition_client.AcquisitionClient import AcquisitionClient
from image_acquisition.acquisition_shared.models_v1 import AsyncJobStatusV1

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Run as MlflowRun, Experiment
except ImportError:
    mlflow = None
    MlflowClient = None
    MlflowRun = None

logger = logging.getLogger(__name__)
logging.getLogger("mlflow").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)
logging.getLogger("requests").setLevel(logging.DEBUG)

class PhotographyExperiment(ABC):
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        if mlflow is None or MlflowClient is None:
            raise RuntimeError("mlflow ist nicht installiert. Installiere mit `pip install mlflow`.")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        self.mlflow = mlflow
        self.client = MlflowClient()
        self._run: Optional[ActiveRun] = None

        logger.debug("Initialized experiment with MLflow at %s", mlflow.get_tracking_uri())

    @abstractmethod
    def _run_impl(self, experiment: Experiment, active_run: MlflowRun):
        """Hauptlogik des Experiments implementieren."""
        pass

    def _get_tags_for_run(self) -> Dict[str, Any]:
        return {}

    def _get_run_name(self) -> Optional[str]:
        """Run Name zurückgeben. Bei Bedarf überschrieben"""
        return None

    def run(self):
        """Führt das Experiment aus."""
        start = time.perf_counter()
        try:
            logger.info("Starting experiment run...")
            experiment_created = self.set_experiment(self.experiment_name)
            logger.debug("Experiment registered. About to start run... ")
            self._run = self.start_run(experiment_id=experiment_created.experiment_id, tags=self._get_tags_for_run(), run_name=self._get_run_name())
            logger.debug("Run started %s... ", self.run)
            self._run_impl(experiment_created, self._run)
            logger.info("Experiment run completed.")
        except Exception as e:
            logger.error("Error during experiment run: %s", e)
            raise e
        end = time.perf_counter()
        self.log_metric("experiment_duration_seconds", end - start)
        self.end_run()

    @abstractmethod
    def configure(self, config: dict):
        """Projekt resp. Service-spezifische Konfiguration (z.B. experiment name, params)."""
        pass

    # --- MLflow Helfer / Lifecycle (vibe coded) ---
    def set_experiment(self, name: str):
        """Setzt oder legt ein Experiment an."""
        return self.mlflow.set_experiment(name)

    def get_experiment(self, name: str):
        """Holt Experiment-Metadaten (oder None)."""
        return self.client.get_experiment_by_name(name)

    def start_run(self, run_name: Optional[str] = None, experiment_id: Optional[str] = None, tags: dict[str, Any] = None, nested: bool = False):
        """Startet einen MLflow-Run und speichert ihn intern."""
        self._run = self.mlflow.start_run(run_name=run_name, experiment_id=experiment_id, tags=tags, nested=nested)
        return self._run

    def end_run(self):
        """Beendet den aktiven Run (falls vorhanden)."""
        if self.mlflow.active_run() is not None:
            self.mlflow.end_run()
        self._run = None

    def log_param(self, key: str, value):
        self.mlflow.log_param(key, value)

    def log_metric(self, key: str, value, step: Optional[int] = None):
        if step is None:
            self.mlflow.log_metric(key, float(value))
        else:
            self.mlflow.log_metric(key, float(value), step=step)

    def log_batch_metrics(self, metrics: dict[str, float], step: Optional[int] = None):
        if step is None:
            self.mlflow.log_metrics(metrics)
        else:
            self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        try:
            logger.info("Uploading artifact %s to run %s (artifact_path=%s)", local_path, self._run.info.run_id, artifact_path)
            # bevorzugt: mlflow API (client-side upload -> server handles)
            self.mlflow.log_artifact(local_path, artifact_path)
            logger.info("Artifact uploaded via mlflow.log_artifact")
            return
        except Exception as e:
            logger.warning("mlflow.log_artifact failed: %s", e)