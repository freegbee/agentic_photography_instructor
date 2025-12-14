import logging
from abc import abstractmethod, ABC
from typing import TypeVar, Generic

from training.mlflow_utils import mlflow_logging

logger = logging.getLogger(__name__)

RESULT = TypeVar('RESULT')


class AbstractLoadData(Generic[RESULT], ABC):

    @abstractmethod
    def _load_data_impl(self):
        pass

    @abstractmethod
    def get_result(self) -> RESULT:
        pass

    @mlflow_logging("load_data_duration_seconds")
    def load_data(self):
        self._load_data_impl()
