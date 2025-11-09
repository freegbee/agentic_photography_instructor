from image_acquisition.acquisition_server.prometheus.Metrics import Metrics, init_metrics


class PrometheusMetrics:

    def __init__(self):
        from prometheus_client import CollectorRegistry, gc_collector, platform_collector, process_collector

        self.prometheus_registry = CollectorRegistry()
        # Default collectors registrieren
        gc_collector.GCCollector(registry=self.prometheus_registry)
        platform_collector.PlatformCollector(registry=self.prometheus_registry)
        process_collector.ProcessCollector(registry=self.prometheus_registry)

        self.prometheus_metrics = init_metrics(registry=self.prometheus_registry)

    def get_registry(self):
        return self.prometheus_registry

    def generate_latest(self) -> bytes:
        from prometheus_client.openmetrics.exposition import generate_latest
        return generate_latest(self.prometheus_registry)

    def metrics(self) -> Metrics:
        return self.prometheus_metrics
