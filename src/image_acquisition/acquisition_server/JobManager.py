from __future__ import annotations
from threading import Lock

from image_acquisition.acquisition_server.ImageAcquisitionJob import ImageAcquisitionJob


class JobManager:

    _instance: JobManager | None = None
    _lock: Lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(JobManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "JobManager":
        return cls()

    def __init__(self):
        if getattr(self, "_initted", False):
            return
        self.jobs: dict[str, ImageAcquisitionJob] = {}
        self._initted = True

    def get_running_jobs(self) -> list[ImageAcquisitionJob]:
        return [job for job in self.jobs.values() if job.is_running()]

    def get_finished_jobs(self) -> list[ImageAcquisitionJob]:
        return [job for job in self.jobs.values() if job.is_finished()]

    def get_failed_jobs(self) -> list[ImageAcquisitionJob]:
        return [job for job in self.jobs.values() if job.status == 'FAILED']

    def get_completed_jobs(self) -> list[ImageAcquisitionJob]:
        return [job for job in self.jobs.values() if job.status == 'COMPLETED']

    def get_all_jobs(self) -> list[ImageAcquisitionJob]:
        return list(self.jobs.values())

    def get_active_jobs(self) -> list[ImageAcquisitionJob]:
        return [job for job in self.jobs.values() if not job.is_finished()]

    def remove_job(self, uuid: str):
        if uuid in self.jobs:
            del self.jobs[uuid]
        else:
            raise KeyError(f"Job with UUID {uuid} not found.")

    def remove_finished_jobs(self):
        for job in self.get_finished_jobs():
            del self.jobs[job.uuid]

    def add_job(self, job: ImageAcquisitionJob):
        if job.uuid in self.jobs:
            raise KeyError(f"Job with UUID {job.uuid} already exists.")
        for j in self.get_active_jobs():
            if j.dataset_id == job.dataset_id:
                raise ValueError(f"An active job for dataset_id {job.dataset_id} already exists with UUID {j.uuid}.")
        self.jobs[job.uuid] = job

    def get_job(self, uuid: str) -> ImageAcquisitionJob:
        if uuid not in self.jobs:
            raise KeyError(f"Job with UUID {uuid} not found.")
        return self.jobs[uuid]