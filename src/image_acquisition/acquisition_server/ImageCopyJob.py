import os
from pathlib import Path
from typing import Optional

from image_acquisition.acquisition_server.abstract_image_job import AbstractImageJob, ServiceResponse
from image_acquisition.acquisition_shared.ImageAcquisitionUtils import ImageAcquisitionUtils
from image_acquisition.acquisition_shared.models_v1 import AsyncImageCopyJobResponseV1


class ImageCopyJob(AbstractImageJob[AsyncImageCopyJobResponseV1]):

    def __init__(self, uuid: str, dataset_id: str, source_path: str, destination_path: str):
        super().__init__(uuid)
        self.dataset_id: Optional[str] = dataset_id
        self.source_path: Optional[Path] = Path(source_path) if source_path is not None else None
        if self.dataset_id is None and self.source_path is None:
            raise ValueError("Either dataset_id or source_path must be provided")
        self.destination_path = Path(destination_path)
        self.resulting_hash: Optional[str] = None
        self.effective_destination_path: Optional[Path] = None

    def start(self):
        effective_source_path = None
        if self.dataset_id:
            dataset_config = self.get_dataset_config(self.dataset_id)
            effective_source_path = dataset_config.calculate_images_root_path()
        else:
            effective_source_path = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.source_path / "images"

        effective_destination_path = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.destination_path / "images"
        ImageAcquisitionUtils.copy_files(effective_source_path, effective_destination_path)
        self.resulting_hash = ImageAcquisitionUtils.compute_dir_hash(effective_destination_path)
        self.effective_destination_path = effective_destination_path

    def create_service_response(self) -> ServiceResponse:
        return AsyncImageCopyJobResponseV1(
                **{"job_uuid": self.uuid, "status": self.status, "resulting_hash": self.resulting_hash, "destination_directory": str(self.effective_destination_path)})



    def is_same_job(self, other_job: 'AbstractImageJob') -> bool:
        if not isinstance(other_job, ImageCopyJob):
            return False
        return self.dataset_id == other_job.dataset_id and self.source_path == other_job.source_path and self.destination_path == other_job.destination_path
