from enum import Enum
from typing import Optional

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
    resulting_hash: str | None = None

class AsyncImageCopyRequestV1(BaseModel):
    source_dataset_id: Optional[str]
    source_directory: Optional[str]
    destination_directory: str

class AsyncImageCopyJobResponseV1(BaseModel):
    job_uuid: str
    status: AsyncJobStatusV1
    destination_directory: str | None = None
    resulting_hash: str | None = None
