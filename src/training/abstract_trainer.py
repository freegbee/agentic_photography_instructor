import logging
import time
from abc import abstractmethod, ABC
from typing import Optional

import mlflow
from mlflow import ActiveRun
from mlflow.entities import Experiment

from training import mlflow_helper
from training.mlflow_helper import MlflowHelper
from training.mlflow_utils import MLflowParamBuilder, MLflowUtils, MLflowTagsBuilder, mlflow_logging

logger = logging.getLogger(__name__)


class AbstractTrainer(ABC):

    def __init__(self, experiment_name: str, source_dataset_id: str):
        self.experiment_name = experiment_name
        self.source_dataset_id = source_dataset_id
        self.mlflow_client = None
        self.run_name = None
        self.experiment: Optional[Experiment] = None
        self._active_run: Optional[ActiveRun] = None
        self.mlflow_helper: Optional[MlflowHelper] = None
        self._init_mlflow()
        logger.info(f"Initialized experiment {self.experiment_name}")

    def _init_mlflow(self):
        self.ml_flow_param_builder = MLflowParamBuilder()
        self.ml_flow_param_builder.add_param("source_dataset_id", self.source_dataset_id)
        self.ml_flow_tags_builder = MLflowTagsBuilder()
        self.mlflow_client = MLflowUtils.get_tracking_client()
        self.experiment = mlflow.set_experiment(self.experiment_name)
        self.mlflow_helper = mlflow_helper

    def run_training(self, run_name: Optional[str] = None):
        self.run_name = run_name
        self._active_run = self.mlflow_helper.start_run(run_name=self.run_name, experiment=self.experiment,
                                                        tags_builder=self.ml_flow_tags_builder)

        MLflowUtils.log_params(self.mlflow_client, self._active_run.info.run_id, self.ml_flow_param_builder)
        MLflowUtils.log_tags(self.mlflow_client, self._active_run.info.run_id, self.ml_flow_tags_builder)

        start_time = time.perf_counter()
        try:
            self._run_training_pipeline()
        finally:
            logger.info("End run %s for experiment %s with duration  %.4f", self.run_name, self.experiment_name,
                        time.perf_counter() - start_time)
            self.mlflow_helper.end_run()
            self._active_run = None

    @mlflow_logging("perf/total_training_duration_seconds")
    def _run_training_pipeline(self):
        logger.info("Start run %s for experiment %s", self.run_name, self.experiment_name)
        self.load_data()
        self.preprocess()
        self.train()
        self.evaluate()
        self.postprocess()

    @mlflow_logging("perf/load_data_duration_seconds")
    def load_data(self):
        self._load_data_impl()

    @mlflow_logging("perf/preprocess_duration_seconds")
    def preprocess(self):
        self._preprocess_impl()

    @mlflow_logging("perf/train_duration_seconds")
    def train(self):
        self._train_impl()

    @mlflow_logging("perf/evaluate_duration_seconds")
    def evaluate(self):
        self._evaluate_impl()

    @mlflow_logging("perf/postprocess_duration_seconds")
    def postprocess(self):
        self._postprocess_impl()

    def log_metric(self, key: str, value, step: Optional[int] = None):
        if step is None:
            self.mlflow_helper.log_metric(key, value)
        else:
            self.mlflow_helper.log_metric(key, value, step=step)

    def log_batch_metrics(self, metrics: dict[str, float], step: Optional[int] = None):
        if step is None:
            self.mlflow_helper.log_batch_metrics(metrics)
        else:
            self.mlflow_helper.log_batch_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        try:
            logger.info("Uploading artifact %s to run %s (artifact_path=%s)", local_path, self._active_run.info.run_id,
                        artifact_path)
            # bevorzugt: mlflow API (client-side upload -> server handles)
            self.mlflow_helper.log_artifact(local_path, artifact_path)
            logger.info("Artifact uploaded via mlflow.log_artifact")
            return
        except Exception as e:
            logger.warning("mlflow.log_artifact failed: %s", e)

    @abstractmethod
    def _load_data_impl(self):
        pass

    @abstractmethod
    def _preprocess_impl(self):
        pass

    @abstractmethod
    def _train_impl(self):
        pass

    @abstractmethod
    def _evaluate_impl(self):
        pass

    def _postprocess_impl(self):
        pass
