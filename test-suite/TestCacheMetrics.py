import threading
import time
from unittest.mock import MagicMock

import numpy as np
from juror_client.juror_cache import JurorCachingService
from juror_shared.models_v1 import ScoringResponsePayloadV1


def test_metrics_hit_miss_counts():
    inner = MagicMock()
    inner.score_ndarray.side_effect = [ScoringResponsePayloadV1(filename='a', score=1.0)]

    cache = JurorCachingService(inner=inner, maxsize=10, ttl=None)

    arr = np.array([1, 2, 3], dtype=np.int32)

    # Erstaufruf -> Miss
    r1 = cache.score_ndarray(arr)
    # Zweitaufruf -> Hit
    r2 = cache.score_ndarray(arr)

    metrics = cache.get_metrics()
    assert metrics['misses'] == 1
    assert metrics['hits'] == 1
    assert metrics['size'] == 1
    assert r1.score == r2.score


class SlowService:
    def __init__(self, delay=0.1):
        self.delay = delay
        self.calls = 0

    def score_ndarray(self, array, filename=None, encoding='npy'):
        self.calls += 1
        time.sleep(self.delay)
        return ScoringResponsePayloadV1(filename=filename, score=float(self.calls))

    def close(self):
        pass


def test_metrics_inflight_dedupe_counts():
    inner = SlowService(delay=0.05)
    cache = JurorCachingService(inner=inner, maxsize=10, ttl=None)

    arr = np.array([9, 9, 9], dtype=np.int32)
    results = []

    def worker():
        res = cache.score_ndarray(arr)
        results.append(res.score)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    metrics = cache.get_metrics()
    # Nur eine miss (die ausführende Anfrage), drei Hits von wartenden Threads
    assert metrics['misses'] == 1
    assert metrics['hits'] == 3
    assert metrics['size'] == 1
    # inner.calls sollte 1 bestätigt
    assert inner.calls == 1
    # Alle Ergebnisse sollten gleich
    assert len(set(results)) == 1

