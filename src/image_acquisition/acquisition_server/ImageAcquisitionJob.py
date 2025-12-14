import logging
import time

from prometheus_client import CollectorRegistry

from image_acquisition.acquisition_server.abstract_image_job import AbstractImageJob, ServiceResponse
from image_acquisition.acquisition_server.handlers.HandlerFactory import HandlerFactory
from image_acquisition.acquisition_server.prometheus import prometheus_metrics
from image_acquisition.acquisition_server.prometheus.Metrics import init_metrics
from image_acquisition.acquisition_shared.models_v1 import AsyncJobStatusV1, AsyncImageAcquisitionJobResponseV1

logger = logging.getLogger(__name__)


class ImageAcquisitionJob(AbstractImageJob[AsyncImageAcquisitionJobResponseV1]):
    uuid: str
    dataset_id: str
    status: AsyncJobStatusV1

    def __init__(self, uuid: str, dataset_id: str):
        logger.info("Initializing ImageAcquisitionJob with uuid=%s for dataset_id=%s", uuid, dataset_id)
        super().__init__(uuid)
        self.dataset_id = dataset_id
        self.resulting_hash = None

        # Load configuration
        dataset_config = self.get_dataset_config(self.dataset_id)
        self.handler = HandlerFactory.create(dataset_config)
        prometheus_registry = CollectorRegistry()
        self.prometheus_metrics = init_metrics(registry=prometheus_registry)

    def start(self):
        start = time.perf_counter()
        self.set_status_running()
        try:
            self.resulting_hash = self.handler.process()
            self.set_status_completed()
        except Exception as e:
            self.set_status_failed()
        finally:
            elapsed = time.perf_counter() - start
            logger.info("ImageAcquisitionJob with %s for %s status is %s after %f seconds with resulting hash %s",
                        self.uuid, self.dataset_id,
                        self.status, elapsed, self.resulting_hash)
            try:
                prometheus_metrics.metrics().IMAGE_ACQUISITION_JOB_DURATION.labels(dataset_id=self.dataset_id,
                                                                                   outcome=self.status).observe(elapsed)
            except Exception as e:
                logger.error(f"Error recording job duration metric: {e}")

    def create_service_response(self) -> ServiceResponse:
        return AsyncImageAcquisitionJobResponseV1(
            **{"job_uuid": self.uuid, "status": self.status, "resulting_hash": self.resulting_hash})

    def is_same_job(self, other_job: 'AbstractImageJob') -> bool:
        if not isinstance(other_job, ImageAcquisitionJob):
            return False
        return self.dataset_id == other_job.dataset_id
