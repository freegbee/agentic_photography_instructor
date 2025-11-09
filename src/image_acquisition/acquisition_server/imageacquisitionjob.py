import logging
import os
import time
from typing import Dict

from prometheus_client import CollectorRegistry

from image_acquisition.acquisition_server.handlers.HandlerFactory import HandlerFactory
from image_acquisition.acquisition_server.prometheus import prometheus_metrics
from image_acquisition.acquisition_server.prometheus.Metrics import init_metrics
from image_acquisition.acquisition_shared.ImageDatasetConfiguration import ImageDatasetConfiguration
from image_acquisition.acquisition_shared.models_v1 import AsyncJobStatusV1
from utils.ConfigLoader import ConfigLoader

logger = logging.getLogger(__name__)


class ImageAcquisitionJob:
    uuid: str
    dataset_id: str
    status: AsyncJobStatusV1

    def __init__(self, uuid: str, dataset_id: str):
        logger.info("Initializing ImageAcquisitionJob with uuid=%s for dataset_id=%s", uuid, dataset_id)
        self.uuid = uuid
        self.dataset_id = dataset_id
        self.status = AsyncJobStatusV1.NEW
        self.resulting_hash = None

        # Load configuration
        config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
        if not config:
            self.status = AsyncJobStatusV1.FAILED
            logger.warning("No config file found in environment %s", os.environ["ENV_NAME"])
            raise ValueError("Kein Konfigurationsobjekt geladen")

        image_acq = config.get('image_acquisition')
        if dataset_id not in image_acq:
            logger.warning("No config found for %s", dataset_id)
            logger.warning("config root %s", image_acq)
            self.status = AsyncJobStatusV1.FAILED
            raise ValueError(f"No configuration found for dataset_id: {dataset_id}")

        config_dict = config['image_acquisition'][dataset_id]
        dataset_config = ImageDatasetConfiguration.from_dict(config_dict)
        self.handler = HandlerFactory.create(dataset_config)
        prometheus_registry = CollectorRegistry()
        self.prometheus_metrics = init_metrics(registry=prometheus_registry)

    def start(self):
        start = time.perf_counter()
        self.status = AsyncJobStatusV1.RUNNING
        try:
            self.resulting_hash = self.handler.process()
            self.status = AsyncJobStatusV1.COMPLETED
        except Exception as e:
            self.status = AsyncJobStatusV1.FAILED
        finally:
            elapsed = time.perf_counter() - start
            logger.info("ImageAcquisitionJob with %s for %s status is %s after %f seconds with resulting hash %s", self.uuid, self.dataset_id,
                        self.status, elapsed, self.resulting_hash)
            try:
                prometheus_metrics.metrics().IMAGE_ACQUISITION_JOB_DURATION.labels(dataset_id=self.dataset_id,
                                                                                   outcome=self.status).observe(elapsed)
            except Exception as e:
                logger.error(f"Error recording job duration metric: {e}")

    def is_running(self):
        return self.status == AsyncJobStatusV1.RUNNING

    def is_finished(self):
        return self.status in {AsyncJobStatusV1.COMPLETED, AsyncJobStatusV1.FAILED}
