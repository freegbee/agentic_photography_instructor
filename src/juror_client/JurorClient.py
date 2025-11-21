"""Kleiner HTTP-Client für den Juror-Scoring-Endpoint.

Bietet eine Klasse `JurorClient` mit Methoden um ein Bild per HTTP POST an
http://localhost:5010/scoring zu schicken.
"""
import base64
import io
import logging
import os
from typing import Optional, Union, BinaryIO, cast

import httpx
import numpy as np

from juror_shared.models_v1 import ScoringRequestPayloadV1, ScoringResponsePayloadV1

logger = logging.getLogger(__name__)


class JurorClient:
    """Client zum Senden von Bildern an den Juror-Server (/scoring).

    Beispiel:
        client = JurorClient()
        result = client.score_image("/path/to/image.jpg")
    """

    def __init__(
            self,
            # FIXME: Die URL ist hier noch hardcoded, sollte aber konfigurierbar sein
            base_url: str = "http://localhost:5010",
            timeout: float = 10.0,
            client: Optional[httpx.Client] = None, ):
        """Erzeuge einen neuen Client.

        Args:
            base_url: Basis-URL des Juror-Servers (Standard: http://localhost:5010)
            timeout: HTTP-Timeout in Sekunden für Requests
            client: Client zum Starten von asynchronen Image Acquisition
        """

        # Validierungen der Parameter
        if base_url is None:
            raise ValueError(
                "base_url darf nicht None sein. Beispiel: 'http://localhost:5010' - Abhängig den der lokalen Konfiguration oder Umgebungsvariable ab.")
        if not isinstance(base_url, str):
            raise TypeError("base_url muss vom Typ str sein")

        self._api_version = "v1"
        self.base_url = base_url.rstrip("/") + f"/{self._api_version}/"  # trailing slash preserves path as directory
        self.scoring_endpoint = "score"

        self.timeout = float(timeout)

        # Connection pooling mit einem persistent httpx.Client
        # Falls ein externer Client übergeben wurde, diesen verwenden (und dann auch nicht schliessen im closer / im __exit__)
        self._external_client = client is not None
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        if client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, limits=limits)
        else:
            self._client = client

    def score_image(self, image_path: str) -> Union[ScoringResponsePayloadV1, str]:
        """Sende eine Bilddatei als Base64-JSON an /score.

        Args:
            image_path: Pfad zur Bilddatei auf dem Dateisystem.

        Returns:
            Das vom Server zurückgegebene JSON (parsed) oder Text, falls kein JSON.

        Raises:
            FileNotFoundError: Wenn die Bilddatei nicht existiert.
            httpx.HTTPError: Bei HTTP/Netzwerkfehlern.
        """

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        url = f"{self.base_url}/score"

        logger.debug("Posting file %s to %s", image_path, url)
        with open(image_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode("ascii")
            payload = ScoringRequestPayloadV1(filename=os.path.basename(image_path), b64=b64)
            json_payload = payload.model_dump()
            logger.debug("Sending image for scoring: %s", json_payload)
            resp = self._client.post(self.scoring_endpoint, json=json_payload)

        # Raise for HTTP errors (4xx/5xx)
        resp.raise_for_status()

        # Versuche JSON zurückzugeben, fallback auf Text
        try:
            return ScoringResponsePayloadV1(**resp.json())
        except ValueError:
            return resp.text

    def score_ndarray(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy") -> Union[
        ScoringResponsePayloadV1, str]:
        """Sende ein numpy.ndarray effizient an den Server via multipart/form-data.

        Das Array wird als .npy (oder .npz wenn encoding == 'npz') in den Request-Body geschrieben.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array muss ein numpy.ndarray sein")

        # Serialisiere das Array in Bytes
        bio = None
        if encoding == "npy":
            bio = io.BytesIO()
            np.save(cast(BinaryIO, bio), array, allow_pickle=False)  # type: ignore[arg-type]
            bio.seek(0)
            filename = filename or "array.npy"
            content_type = "application/octet-stream"
        elif encoding == "npz":
            bio = io.BytesIO()
            np.savez(cast(BinaryIO, bio), array=array)  # type: ignore[arg-type]
            bio.seek(0)
            filename = filename or "array.npz"
            content_type = "application/octet-stream"
        else:
            raise ValueError("Unsupported encoding; use 'npy' or 'npz'")

        files = {"array_file": (filename, bio.read(), content_type)}
        logger.info(f"Sending file for scoring: %s", files["array_file"][0])
        logger.debug("Sending array file for scoring: %s", filename)
        resp = self._client.post(self.scoring_endpoint + "/ndarray", files=files)
        logger.debug("Response status=%s headers=%s", resp.status_code, dict(resp.headers))

        resp.raise_for_status()

        try:
            return ScoringResponsePayloadV1(**resp.json())
        except ValueError as e:
            logger.exception(e)
            return resp.text

    def close(self) -> None:
        """Schliesse den HTTP-Client, falls dieser intern erstellt wurde."""
        if not self._external_client:
            try:
                self._client.close()
            except Exception:
                logger.exception("Fehler beim Schliessen des HTTP-Clients")

    def __enter__(self) -> "JurorClient":
        """Context-Manager entry: Wird verwendet im Rahmen des Python-Context-Manager-Protocol (aka with-Anweisung)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context-Manager exit: Wird beim Verlassen der with-Anweisung verwendet und schliesst den httpx client"""
        self.close()
