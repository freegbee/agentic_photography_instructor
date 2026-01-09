import logging
import uuid
from typing import Dict

from data_types.ImageDatasetConfiguration import ImageDatasetConfiguration
from image_acquisition.acquisition_client.AcquisitionClient import AcquisitionClient
from image_acquisition.acquisition_shared.models_v1 import AsyncImageAcquisitionJobResponseV1, AsyncJobStatusV1
from utils.ConfigLoader import ConfigLoader

logger = logging.getLogger(__name__)

class DummyAcquisitionClient(AcquisitionClient):
    """
    Ein Dummy-Client für Optimierungsläufe (z.B. Optuna), der die Bildakquise simuliert.
    Er umgeht die tatsächliche Hash-Berechnung und den Download und liefert sofortigen Erfolg
    basierend auf den Konfigurationswerten zurück.
    """

    def __init__(self):
        super().__init__("http://localhost:5005")
        logger.warning("Using dummy acquisition client. No validation or downloading of dataset. You have to have it :-)")
        self._active_jobs: Dict[str, str] = {}


    def start_async_image_acquisition(self, dataset_id: str) -> str:
        """
        Simuliert den Start eines Akquise-Jobs, indem eine UUID generiert und die dataset_id gespeichert wird.
        """
        job_uuid = str(uuid.uuid4())
        self._active_jobs[job_uuid] = dataset_id
        return job_uuid

    def query_async_image_acquisition_job(self, job_uuid: str) -> AsyncImageAcquisitionJobResponseV1:
        """
        Liefert sofortigen Erfolg mit dem Hash aus der Konfiguration für die gegebene Job-UUID.
        """
        if job_uuid not in self._active_jobs:
            return AsyncImageAcquisitionJobResponseV1(
                **{"job_uuid": job_uuid, "status": AsyncJobStatusV1.FAILED, "resulting_hash": None})

        dataset_id = self._active_jobs[job_uuid]

        try:
            config_dict = ConfigLoader.load_dataset_config(dataset_id)
            dataset_config = ImageDatasetConfiguration.from_dict(dataset_id, config_dict)
            target_hash = dataset_config.target_hash

            if not target_hash:
                return AsyncImageAcquisitionJobResponseV1(
                    **{"job_uuid": job_uuid, "status": AsyncJobStatusV1.FAILED, "resulting_hash": None})

            return AsyncImageAcquisitionJobResponseV1(
                **{"job_uuid": job_uuid, "status": AsyncJobStatusV1.COMPLETED, "resulting_hash": target_hash})

        except Exception as e:
            return AsyncImageAcquisitionJobResponseV1(
                **{"job_uuid": job_uuid, "status": AsyncJobStatusV1.FAILED, "resulting_hash": None})
