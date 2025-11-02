"""Kleiner HTTP-Client für den Juror-Scoring-Endpoint.

Bietet eine Klasse `JurorClient` mit Methoden um ein Bild per HTTP POST an
http://localhost:5010/scoring zu schicken.
"""
import os
import logging
import base64
from typing import Optional

import httpx

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
            base_url: str = "http://localhost:5010",
            timeout: int = 10,
            client: Optional[httpx.Client] = None,):
        """Erzeuge einen neuen Client.

        Args:
            base_url: Basis-URL des Juror-Servers (Standard: http://localhost:5010)
            timeout: HTTP-Timeout in Sekunden für Requests
        """
        self._api_version = "v1"
        self.base_url = base_url.rstrip("/") + f"/{self._api_version}"
        self.timeout = timeout

        # Connection pooling mit einem persistent httpx.Client
        # Falls ein externer Client übergeben wurde, diesen verwenden (und dann auch nicht schliessen im closer / im __exit__)
        self._external_client = client is not None
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        if client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, limits=limits)
        else:
            self._client = client

    def score_image(self, image_path: str) -> ScoringResponsePayloadV1 | str:
        """Sende eine Bilddatei als Base64-JSON an /score_base64.

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

        logger.debug("Posting file %s to %s (mime=%s)", image_path, url)
        with open(image_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode("ascii")
            payload = ScoringRequestPayloadV1(filename=image_path, b64=b64)
            json_payload = payload.model_dump()
            logger.debug("Sending image for scoring:", json_payload)
            resp = httpx.post(url, json=json_payload, timeout=self.timeout, headers={"Content-Type": "application/json"})

        # Raise for HTTP errors (4xx/5xx)
        resp.raise_for_status()

        # Versuche JSON zurückzugeben, fallback auf Text
        try:
            return ScoringResponsePayloadV1(**resp.json());
        except ValueError:
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