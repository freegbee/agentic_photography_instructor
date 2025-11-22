import threading
import time

import numpy as np

from juror_client.juror_cache import JurorCachingService
from juror_client.juror_service import JurorService
from juror_shared.models_v1 import ScoringResponsePayloadV1


class SlowMockService(JurorService):
    def __init__(self, delay=0.5):
        self.calls = 0
        self.delay = delay

    def score_image(self, image_path: str):
        raise NotImplementedError

    def score_ndarray(self, array, filename=None, encoding: str = "npy"):
        # Simuliere eine langsame Verarbeitung
        self.calls += 1
        time.sleep(self.delay)
        return ScoringResponsePayloadV1(filename=filename, score=float(self.calls))

    def close(self):
        pass


def test_cache_hit_and_miss():
    inner = SlowMockService(delay=0.01)
    cache = JurorCachingService(inner=inner, maxsize=10, ttl=None)

    arr = np.array([1, 2, 3], dtype=np.int32)

    r1 = cache.score_ndarray(arr)
    r2 = cache.score_ndarray(arr)

    assert inner.calls == 1
    assert r1.score == r2.score


def test_in_flight_dedupe():
    inner = SlowMockService(delay=0.1)
    cache = JurorCachingService(inner=inner, maxsize=10, ttl=None)

    arr = np.array([9, 8, 7], dtype=np.int32)
    results = []

    def worker():
        res = cache.score_ndarray(arr)
        results.append(res.score)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    duration = time.time() - start

    # Inner should have been called only once due to dedupe
    assert inner.calls == 1
    # All results should be equal
    assert len(set(results)) == 1
    # Duration should be roughly the service delay, not 3x the delay
    assert duration < 0.5


def test_ttl_expiration():
    inner = SlowMockService(delay=0.01)
    cache = JurorCachingService(inner=inner, maxsize=10, ttl=0.05)

    arr = np.array([4, 5, 6], dtype=np.int32)

    r1 = cache.score_ndarray(arr)
    assert inner.calls == 1
    # Wait beyond ttl
    time.sleep(0.06)
    r2 = cache.score_ndarray(arr)
    assert inner.calls == 2
    assert r1.score != r2.score

