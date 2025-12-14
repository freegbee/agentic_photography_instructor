import asyncio
import logging
import os
from typing import Dict

from image_acquisition.acquisition_client.AcquisitionClient import AcquisitionClient
from image_acquisition.acquisition_shared.models_v1 import AsyncJobStatusV1
from utils.ConfigLoader import ConfigLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def main():
    config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
    await start_job()


async def start_job():
    with AcquisitionClient(base_url=os.environ["IMAGE_ACQUISITION_SERVICE_URL"]) as acquisition_client:
        job_uuid = acquisition_client.start_async_image_acquisition("single_image")
        #job_uuid = acquisition_client.start_async_image_acquisition("places_365_split_two_actions")
        #job_uuid = acquisition_client.start_async_image_acquisition("places365_val_large")
        logger.info(f"Started async image acquisition job with UUID: {job_uuid}")
        sleep_interval_seconds = 5
        wait_time_seconds = 60 * 15
        max_polling = int(wait_time_seconds / sleep_interval_seconds)
        for i in range(max_polling):  # Check status for up to wait_time_seconds seconds with a sleep interval of sleep_interval_seconds
            await asyncio.sleep(sleep_interval_seconds)
            job_status = acquisition_client.query_async_image_acquisition_job(job_uuid)
            if job_status is not None:
                logger.info("Job status at attempt %s: %s; hash %s", i, job_status.status, job_status.resulting_hash)
                if job_status.status == AsyncJobStatusV1.COMPLETED:
                    logger.info(f"Job completed successfully.")
                    break
            else:
                logger.info(f"Job with UUID {job_uuid} not found.")


if __name__ == "__main__":
    asyncio.run(main())