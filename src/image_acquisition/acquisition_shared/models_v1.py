from enum import Enum

from pydantic import BaseModel


class AsyncJobStatusV1(Enum):
    NEW = "NEW"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class StartAsyncImageAcquisitionRequestV1(BaseModel):
    dataset_id: str

class AsyncImageAcquisitionJobResponseV1(BaseModel):
    job_uuid: str
    status: AsyncJobStatusV1

