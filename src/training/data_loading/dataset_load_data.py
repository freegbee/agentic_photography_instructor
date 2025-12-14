import logging
import os
import time
from typing import Dict, Optional

from data_types.ImageDatasetConfiguration import ImageDatasetConfiguration
from image_acquisition.acquisition_client.AcquisitionClient import AcquisitionClient
from image_acquisition.acquisition_shared.models_v1 import AsyncJobStatusV1
from training import mlflow_helper
from training.data_loading.abstract_load_data import AbstractLoadData, RESULT
from utils.ConfigLoader import ConfigLoader

logger = logging.getLogger(__name__)


class DatasetLoadDataResult:
    def __init__(self, image_dataset_hash: str):
        self.image_dataset_hash = image_dataset_hash


class DatasetLoadData(AbstractLoadData[DatasetLoadDataResult]):

    def __init__(self, dataset_id: str, acquisition_client: AcquisitionClient = None):
        super().__init__()
        self.dataset_id = dataset_id
        self.acquisition_client = acquisition_client
        self.image_dataset_hash: Optional[str] = None

    def _load_data_impl(self):
        try:
            config_dict: Dict = ConfigLoader.load_dataset_config(self.dataset_id)
        except Exception as e:
            logger.exception("Exception loading config for dataset %s: %s", self.dataset_id, e)
            raise e
        # Dataset-Konfiguration laden und Bildpfad ermitteln
        self.dataset_config = ImageDatasetConfiguration.from_dict(self.dataset_id, config_dict)
        self.image_dataset_hash = self._ensure_image_dataset(self.dataset_config.dataset_id)
        mlflow_helper.log_param("dataset_hash", self.image_dataset_hash)

    def get_result(self) -> RESULT:
        return DatasetLoadDataResult(self.image_dataset_hash)

    def _ensure_image_dataset(self, dataset_id: str) -> str:
        """Stellt sicher, dass der Bild-Datensatz geladen ist (Platzhalter)."""
        result_hash = None
        if self.acquisition_client:
            result_hash = self._ensure_via_acquisition_client(self.acquisition_client, self.dataset_id)
        else:
            with AcquisitionClient(base_url=os.environ["IMAGE_ACQUISITION_SERVICE_URL"]) as acquisition_client:
                result_hash = self._ensure_via_acquisition_client(acquisition_client, self.dataset_id)
        return result_hash

    def _ensure_via_acquisition_client(self, acquisition_client: AcquisitionClient, dataset_id: str) -> str | None:
        """Stellt sicher, dass der Bild-Datensatz geladen ist, indem der acquisition_client verwendet wird."""
        job_uuid = acquisition_client.start_async_image_acquisition(dataset_id)
        logger.info("Started async image acquisition job with UUID %s for dataset %s", job_uuid, dataset_id)
        # Warten auf Abschluss (vereinfachtes Polling)
        while True:
            job_status = acquisition_client.query_async_image_acquisition_job(job_uuid)
            if job_status is not None:
                logger.debug("Job status: %s", job_status.status)
                if job_status.status == AsyncJobStatusV1.COMPLETED:
                    logger.info("Image dataset %s acquired successfully. Resulting dataset hash: %s", dataset_id,
                                job_status.resulting_hash)
                    result_hash = job_status.resulting_hash
                    break
                if job_status.status == AsyncJobStatusV1.FAILED:
                    logger.error("Image acquisition job %s failed.", job_uuid)
                    raise RuntimeError(f"Image acquisition job {job_uuid} failed.")
            else:
                logger.warning("Job with UUID %s not found.", job_uuid)
            import time
            time.sleep(5)  # Wartezeit zwischen den Abfragen
        return result_hash
