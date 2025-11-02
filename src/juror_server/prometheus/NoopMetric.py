# Prometheus client (mit Fallback, falls nicht installiert)
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
except Exception:
    # Minimaler No-op Ersatz, um Abstürze zu vermeiden wenn das Paket fehlt
    class _NoopMetric:
        def __init__(self, *a, **k): pass
        def labels(self, *a, **k): return self
        def inc(self, *a, **k): pass
        def set(self, *a, **k): pass
        def observe(self, *a, **k): pass
        def time(self):
            class _Ctx:
                def __enter__(self): pass
                def __exit__(self, *e): pass
            return _Ctx()
    Counter = Histogram = Gauge = _NoopMetric
    def generate_latest(): return b""
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"