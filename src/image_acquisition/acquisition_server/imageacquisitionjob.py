from image_acquisition.acquisition_shared.models_v1 import AsyncJobStatusV1


class ImageAcquisitionJob:
    uuid: str
    dataset_id: str
    status: AsyncJobStatusV1

    def __init__(self, uuid: str, dataset_id: str):
        self.uuid = uuid
        self.dataset_id = dataset_id
        self.status = AsyncJobStatusV1.NEW