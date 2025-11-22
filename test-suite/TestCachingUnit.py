import numpy as np
from unittest.mock import MagicMock

import pytest

from juror_client.juror_cache import JurorCachingService
from juror_shared.models_v1 import ScoringResponsePayloadV1


def test_caching_service_uses_inner_only_once_on_duplicate_calls():
    inner = MagicMock()
    # inner.score_ndarray returns different values based on call count
    inner.score_ndarray.side_effect = [ScoringResponsePayloadV1(filename='a', score=1.0), ScoringResponsePayloadV1(filename='a', score=2.0)]

    cache = JurorCachingService(inner=inner, maxsize=10, ttl=None)

    arr = np.array([1, 2, 3], dtype=np.int32)

    r1 = cache.score_ndarray(arr)
    r2 = cache.score_ndarray(arr)

    # inner should only have been called once
    assert inner.score_ndarray.call_count == 1
    assert r1.score == r2.score


def test_caching_service_handles_exceptions_and_propagates():
    inner = MagicMock()
    def raise_error(array, filename=None, encoding='npy'):
        raise RuntimeError("boom")
    inner.score_ndarray.side_effect = raise_error

    cache = JurorCachingService(inner=inner, maxsize=10, ttl=None)
    arr = np.array([9, 9, 9], dtype=np.int32)

    try:
        cache.score_ndarray(arr)
        pytest.fail("expected exception")
    except RuntimeError:
        pass

    # subsequent call should try again (not cache the exception permanently)
    inner.score_ndarray.side_effect = [ScoringResponsePayloadV1(filename='a', score=3.0)]
    r = cache.score_ndarray(arr)
    assert r.score == 3.0
