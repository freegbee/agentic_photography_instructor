from prometheus_client import Counter, Summary, Gauge, Histogram

# Prometheus Metriken
REQUEST_COUNT = Counter(
    "juror_requests_total",
    "Total HTTP requests to Juror service",
    ['method', 'endpoint', 'status']
)
SCORING_DURATION = Histogram(
    "juror_scoring_duration_seconds",
    "Duration to score an image (seconds)"
)
JUROR_LOADED = Gauge(
    "juror_model_loaded",
    "1 if juror model is loaded, 0 otherwise"
)
ERROR_COUNT = Counter(
    "juror_errors_total",
    "Total inference errors",
    ['type']
)