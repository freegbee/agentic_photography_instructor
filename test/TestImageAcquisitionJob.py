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
    with AcquisitionClient() as acquisition_client:
        job_uuid = acquisition_client.start_async_image_acquisition("five_images")
        logger.info(f"Started async image acquisition job with UUID: {job_uuid}")
        for i in range(3):
            await asyncio.sleep(3)
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