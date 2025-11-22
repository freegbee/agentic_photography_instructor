import numpy as np

from juror_client.juror_cache import _hash_ndarray


def test_hash_same_content_equal():
    a = np.array([[1, 2, 3]], dtype=np.int32)
    b = np.array([[1, 2, 3]], dtype=np.int32)

    h1 = _hash_ndarray(a)
    h2 = _hash_ndarray(b)

    assert isinstance(h1, str)
    assert h1 == h2


def test_hash_different_shape_or_dtype_different():
    a = np.array([[1, 2, 3]], dtype=np.int32)
    b = np.array([1, 2, 3], dtype=np.int32)  # different shape
    c = np.array([[1, 2, 3]], dtype=np.int16)  # different dtype

    assert _hash_ndarray(a) != _hash_ndarray(b)
    assert _hash_ndarray(a) != _hash_ndarray(c)


def test_hash_sensitive_to_content_change():
    a = np.array([[1, 2, 3]], dtype=np.int32)
    d = np.array([[1, 2, 4]], dtype=np.int32)

    assert _hash_ndarray(a) != _hash_ndarray(d)

