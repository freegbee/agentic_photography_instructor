from prometheus_client import Histogram


class Metrics:
    def __init__(self, registry):
        self.HTTP_REQUEST_DURATION = Histogram(name="imageacquisition_request_duration_seconds",
                                               documentation="HTTP request duration in seconds",
                                               labelnames=['method', 'endpoint', 'status'],
                                               unit="seconds",
                                               buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
                                               registry=registry)
        # Histogram for job durations (seconds) with labels for dataset, outcome and uuid
        self.IMAGE_ACQUISITION_JOB_DURATION = Histogram(name="image_acquisition_job_duration_seconds",
                                                        documentation="Duration of image acquisition job in seconds",
                                                        labelnames=["dataset_id", "outcome"],
                                                        unit="seconds",
                                                        buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
                                                        registry=registry)


def init_metrics(registry):
    return Metrics(registry)
