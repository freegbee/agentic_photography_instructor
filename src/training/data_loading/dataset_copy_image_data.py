from typing import Optional

from image_acquisition.acquisition_client.AcquisitionClient import AcquisitionClient
from training.data_loading.abstract_load_data import AbstractLoadData, RESULT


class DatasetCopyImageDataResult:
    def __init__(self, copied_image_directory: str):
        self.copied_image_directory = copied_image_directory


class DatasetCopyImageData(AbstractLoadData):

    def __init__(self, dataset_id: str, acquisition_client: AcquisitionClient = None):
        super().__init__()
        self.dataset_id = dataset_id
        self.acquisition_client = acquisition_client
        self.target_directory: Optional[str] = None

    def _load_data_impl(self):
        pass

    def get_result(self) -> RESULT:
        return DatasetCopyImageDataResult(self.target_directory)
