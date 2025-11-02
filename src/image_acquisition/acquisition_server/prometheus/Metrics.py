from prometheus_client import Histogram


class Metrics:
    def __init__(self, registry):
        self.HTTP_REQUEST_DURATION = Histogram(name="imageacquisition_request_duration_seconds",
                                               documentation="HTTP request duration in seconds",
                                               labelnames=['method', 'endpoint', 'status'],
                                               unit="seconds",
                                               buckets=(0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2.5,5,10),
                                               registry=registry)

def init_metrics(registry):
    return Metrics(registry)