import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TypeVar, Generic

from mlflow import MlflowClient

from training.mlflow_helper import MlflowHelper
from training.mlflow_utils import mlflow_logging

logger = logging.getLogger(__name__)

RESULT = TypeVar('RESULT')


class AbstractPreprocessor(Generic[RESULT], ABC):

    def __init__(self, preprocessing_step: Optional[int]) -> None:
        self.step_arg = preprocessing_step

    @abstractmethod
    def get_preprocessing_result(self) -> RESULT:
        pass

    @abstractmethod
    def _preprocess_impl(self):
        pass

    @mlflow_logging("preprocessing_duration_seconds")
    def preprocess(self) -> RESULT:
        self._preprocess_impl()
        return self.get_preprocessing_result()

    @staticmethod
    def get_image_root_path() -> Path:
        return Path(os.environ["IMAGE_VOLUME_PATH"])