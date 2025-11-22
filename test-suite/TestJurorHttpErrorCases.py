import httpx
import pytest
import numpy as np
from pathlib import Path

from juror_client.juror_service import JurorHttpService
from juror_shared.models_v1 import ScoringResponsePayloadV1


def test_server_error_raises_http_status_error_for_image(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server failure"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver/v1/")
    service = JurorHttpService(base_url="http://testserver", client=client)

    tmp_file = tmp_path / "nonexistent.jpg"
    # create a small temp file for the call
    tmp_file.write_bytes(b"DATA")

    with pytest.raises(httpx.HTTPStatusError):
        service.score_image(str(tmp_file))

    tmp_file.unlink()


def test_server_error_raises_http_status_error_for_ndarray():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"internal error")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver/v1/")
    service = JurorHttpService(base_url="http://testserver", client=client)

    arr = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(httpx.HTTPStatusError):
        service.score_ndarray(arr)


def test_non_json_response_returns_text_for_ndarray():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"NOT JSON")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver/v1/")
    service = JurorHttpService(base_url="http://testserver", client=client)

    arr = np.array([7, 8, 9], dtype=np.int32)
    resp = service.score_ndarray(arr)
    assert isinstance(resp, str)
    assert resp == "NOT JSON"


def test_transport_error_propagates_for_ndarray():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver/v1/")
    service = JurorHttpService(base_url="http://testserver", client=client)

    arr = np.array([10, 11, 12], dtype=np.int32)
    with pytest.raises(httpx.ConnectError):
        service.score_ndarray(arr)


def test_timeout_error_propagates_for_image(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timeout")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver/v1/")
    service = JurorHttpService(base_url="http://testserver", client=client)

    file_path = tmp_path / "img_timeout.jpg"
    file_path.write_bytes(b"DATA")

    with pytest.raises(httpx.ReadTimeout):
        service.score_image(str(file_path))
