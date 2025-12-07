import numpy as np
from juror_client.JurorClient import JurorClient
from juror_client.registry import clear_registry, get_registered_service
from juror_client.juror_cache import JurorCachingService


class FakeResponse:
    def __init__(self, json_obj):
        self._json = json_obj
        self.status_code = 200
        self.headers = {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHttpClient:
    def __init__(self, score_value=0.5):
        self.post_count = 0
        self.score_value = score_value

    def post(self, endpoint, files=None, json=None):
        # Simulate a response payload similar to the real juror
        self.post_count += 1
        return FakeResponse({"score": float(self.score_value)})


def test_jurorclient_uses_caching_service_and_caches_ndarray_calls():
    # Ensure clean registry so get_juror_service will create & register
    clear_registry()

    fake_client = FakeHttpClient(score_value=0.42)

    # Create a JurorClient that will construct a JurorHttpService with our fake client
    # and wrap it in JurorCachingService via get_juror_service(use_cache=True)
    jc = JurorClient(base_url="http://dummy", timeout=1.0, client=fake_client, register_name="jc_test", use_cache=True)

    # Ensure the registry has the cached service registered
    svc = get_registered_service("jc_test")
    assert svc is not None
    assert isinstance(svc, JurorCachingService)

    arr = np.zeros((8, 8, 3), dtype=np.uint8)

    # First call should be a miss -> underlying HTTP post is invoked
    resp1 = jc.score_ndarray_rgb(arr)
    # Second call with identical content should hit the cache
    resp2 = jc.score_ndarray_rgb(arr.copy())

    # Underlying fake http client's post should only have been called once
    assert fake_client.post_count == 1

    # Cache metrics should show 1 miss and 1 hit
    metrics = svc.get_metrics()
    assert metrics["misses"] == 1
    assert metrics["hits"] == 1

    # Results should be equal and contain the scored value
    # resp1 / resp2 are ScoringResponsePayloadV1 instances or dict-like; compare score attribute or dict
    s1 = getattr(resp1, 'score', None) or (resp1.get('score') if isinstance(resp1, dict) else None)
    s2 = getattr(resp2, 'score', None) or (resp2.get('score') if isinstance(resp2, dict) else None)
    assert float(s1) == float(s2) == 0.42

    # cleanup
    try:
        jc.close()
    except Exception:
        pass
    clear_registry()

