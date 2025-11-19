import logging
import os
from typing import List

from dataset.COCODataset import COCODataset
from image_acquisition.acquisition_client.AcquisitionClient import AcquisitionClient
from image_acquisition.acquisition_shared.models_v1 import AsyncJobStatusV1

logger = logging.getLogger(__name__)

class Utils:
    @staticmethod
    def ensure_image_dataset(dataset_id: str) -> str:
        """Stellt sicher, dass der Bild-Datensatz geladen ist (Platzhalter)."""
        result_hash = None
        with AcquisitionClient(base_url=os.environ["IMAGE_ACQUISITION_SERVICE_URL"]) as acquisition_client:
            job_uuid = acquisition_client.start_async_image_acquisition(dataset_id)
            logger.info("Started async image acquisition job with UUID %s for dataset %s", job_uuid, dataset_id)
            # Warten auf Abschluss (vereinfachtes Polling)
            while True:
                job_status = acquisition_client.query_async_image_acquisition_job(job_uuid)
                if job_status is not None:
                    logger.debug("Job status: %s", job_status.status)
                    if job_status.status == AsyncJobStatusV1.COMPLETED:
                        logger.info("Image dataset %s acquired successfully. Resulting dataset hash: %s", dataset_id, job_status.resulting_hash)
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

    @staticmethod
    def calculate_image_root_path(dataset_id: str) -> str:
        """Berechnet den Pfad zum Bildstammverzeichnis basierend auf der dataset_id."""
        base_path = os.environ.get("IMAGE_DATASET_BASE_PATH", "/data/images")
        return os.path.join(base_path, dataset_id)

    @staticmethod
    def collate_keep_size(batch):
        # Returns a batch without stacking the images, so that they can keep their original size
        return batch