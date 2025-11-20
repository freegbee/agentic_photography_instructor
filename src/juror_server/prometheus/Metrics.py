from prometheus_client import Counter, Gauge, Histogram


class Metrics:
    def __init__(self, registry):
        self.HTTP_REQUEST_DURATION = Histogram(name="juror_request_duration_seconds",
                                               documentation="HTTP request duration in seconds",
                                               labelnames=['method', 'endpoint', 'status'],
                                               unit="seconds",
                                               buckets=(0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.0275,0.03,0.0325,0.035,0.0375,0.04,0.05, 0.06,0.08,0.1,0.2,0.5),
                                               registry=registry)
        self.SCORING_DURATION = Histogram(name="juror_scoring_duration_seconds",
                                          documentation="Duration to score an image (seconds)",
                                          unit="seconds",
                                          registry=registry)
        self.JUROR_LOADED = Gauge(name="juror_model_loaded", documentation="1 if juror model is loaded, 0 otherwise",
                                  registry=registry)
        self.ERROR_COUNT = Counter(name="juror_errors_total", documentation="Total inference errors",
                                   labelnames=['type'], registry=registry)
        self.SCORING_VALUE = Gauge(name="juror_scoring_value", documentation="Score score based on juror model",
                                   registry=registry)


def init_metrics(registry):
    return Metrics(registry)
