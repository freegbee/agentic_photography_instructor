import time
import numpy as np
from unittest.mock import MagicMock

from juror_client.juror_cache import JurorCachingService
from juror_shared.models_v1 import ScoringResponsePayloadV1


def make_array(val: int):
    return np.array([val], dtype=np.int32)


def test_lru_eviction_behavior():
    # Deterministischer Test: wir prüfen die Keys über die öffentliche API.
    inner = MagicMock()
    inner.score_ndarray.side_effect = [ScoringResponsePayloadV1(filename='a', score=1.0),
                                       ScoringResponsePayloadV1(filename='b', score=2.0),
                                       ScoringResponsePayloadV1(filename='c', score=3.0)]

    cache = JurorCachingService(inner=inner, maxsize=2, ttl=None)

    a1 = make_array(1)
    a2 = make_array(2)
    a3 = make_array(3)

    # Insert two entries
    _ = cache.score_ndarray(a1)
    _ = cache.score_ndarray(a2)

    # At this point both keys should be present (use public API)
    k1 = cache.make_cache_key(a1, None, 'npy')
    k2 = cache.make_cache_key(a2, None, 'npy')
    keys_now = cache.cached_keys()
    assert k1 in keys_now and k2 in keys_now

    # Access a1 to make it most-recently-used
    _ = cache.score_ndarray(a1)

    # Insert a3, this must evict the least-recently-used key (which is k2)
    _ = cache.score_ndarray(a3)

    k3 = cache.make_cache_key(a3, None, 'npy')
    # Now cache should contain exactly k1 and k3
    keys = cache.cached_keys()
    assert k1 in keys and k3 in keys
    assert k2 not in keys
    # inner should have been called three times (a1,a2,a3)
    assert inner.score_ndarray.call_count == 3


def test_ttl_bounds_expiration():
    inner = MagicMock()
    inner.score_ndarray.side_effect = [ScoringResponsePayloadV1(filename='x', score=1.0),
                                       ScoringResponsePayloadV1(filename='x', score=2.0)]

    # TTL = 0.05 seconds
    ttl = 0.05
    cache = JurorCachingService(inner=inner, maxsize=10, ttl=ttl)

    arr = make_array(99)

    r1 = cache.score_ndarray(arr)
    assert inner.score_ndarray.call_count == 1

    # Immediately (well within TTL) - should be a hit
    time.sleep(ttl / 4)
    r_hit = cache.score_ndarray(arr)
    assert inner.score_ndarray.call_count == 1
    assert r1.score == r_hit.score

    # Wait just past TTL boundary
    time.sleep(ttl + 0.02)
    r2 = cache.score_ndarray(arr)
    # inner should have been called again due to expiration
    assert inner.score_ndarray.call_count >= 2
    assert r2.score != r1.score
