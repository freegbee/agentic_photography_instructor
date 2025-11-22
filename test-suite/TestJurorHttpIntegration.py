import httpx
import pytest
import numpy as np
from pathlib import Path

from juror_client.juror_service import JurorHttpService
from juror_shared.models_v1 import ScoringResponsePayloadV1


# --- Fixtures für wiederverwendbare Mock-Services ---
@pytest.fixture
def ndarray_service() -> JurorHttpService:
    """Service mit Mock-Transport für `score_ndarray`-Anfragen."""

    def handler(request: httpx.Request) -> httpx.Response:
        # Wir erwarten ein POST auf den ndarray-Endpunkt
        assert request.method == "POST"
        assert "score/ndarray" in request.url.path
        payload = {"filename": "array.npy", "score": 0.75}
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver/v1/")
    service = JurorHttpService(base_url="http://testserver", client=client)
    return service


@pytest.fixture
def image_service() -> JurorHttpService:
    """Service mit Mock-Transport für `score_image`-Anfragen."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        # B64-JSON goes to /v1/score
        assert request.url.path.endswith("/v1/score") or request.url.path.endswith("/score")
        body = request.read()
        assert b"b64" in body
        payload = {"filename": "img.jpg", "score": 0.33}
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver/v1/")
    service = JurorHttpService(base_url="http://testserver", client=client)
    return service


@pytest.fixture
def non_json_service() -> JurorHttpService:
    """Service mit Mock-Transport, das keine JSON-Antwort zurückgibt."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"OK")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver/v1/")
    service = JurorHttpService(base_url="http://testserver", client=client)
    return service


# --- Tests ---
def test_score_ndarray_with_mocktransport(ndarray_service: JurorHttpService):
    """Integrationstest: score_ndarray sollte mit einem Mock-HTTP-Transport korrekt antworten."""
    arr = np.array([[1, 2, 3]], dtype=np.int16)
    resp = ndarray_service.score_ndarray(arr)

    assert isinstance(resp, ScoringResponsePayloadV1)
    assert resp.score == 0.75


def test_score_image_with_mocktransport(image_service: JurorHttpService, tmp_path: Path):
    """Integrationstest: score_image sollte eine Datei lesen und das JSON-Resultat zurückgeben."""
    file_path = tmp_path / "img.jpg"
    file_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    resp = image_service.score_image(str(file_path))
    assert isinstance(resp, ScoringResponsePayloadV1)
    assert resp.score == 0.33


def test_non_json_response_returns_text(non_json_service: JurorHttpService, tmp_path: Path):
    """Wenn der Server eine nicht-JSON-Antwort gibt, sollte der Service den Text zurückliefern."""
    file_path = tmp_path / "img2.jpg"
    file_path.write_bytes(b"DATA")

    resp = non_json_service.score_image(str(file_path))
    assert isinstance(resp, str)
    assert resp == "OK"
