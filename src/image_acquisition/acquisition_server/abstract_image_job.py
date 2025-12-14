import logging
from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Generic

from data_types.ImageDatasetConfiguration import ImageDatasetConfiguration
from image_acquisition.acquisition_shared.models_v1 import AsyncJobStatusV1
from utils.ConfigLoader import ConfigLoader

logger = logging.getLogger(__name__)

ServiceResponse = TypeVar('ServiceResponse')

class AbstractImageJob(Generic[ServiceResponse], ABC):
    def __init__(self, uuid: str):
        self.uuid = uuid
        self.status = AsyncJobStatusV1.NEW

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def is_same_job(self, other_job: 'AbstractImageJob') -> bool:
        pass

    @abstractmethod
    def create_service_response(self) -> ServiceResponse:
        pass

    def get_dataset_config(self, dataset_id: str) -> ImageDatasetConfiguration:
        # Load configuration
        try:
            config_dict: Dict = ConfigLoader.load_dataset_config(dataset_id)
        except Exception as e:
            self.set_status_failed()
            logger.exception("Exception loading config for dataset %s: %s", dataset_id, e)
            raise e

        return ImageDatasetConfiguration.from_dict(dataset_id, config_dict)


    def set_status_running(self):
        self.status = AsyncJobStatusV1.RUNNING

    def set_status_completed(self):
        self.status = AsyncJobStatusV1.COMPLETED

    def set_status_failed(self):
        self.status = AsyncJobStatusV1.FAILED

    def is_running(self):
        return self.status == AsyncJobStatusV1.RUNNING

    def is_finished(self):
        return self.status in {AsyncJobStatusV1.COMPLETED, AsyncJobStatusV1.FAILED}
