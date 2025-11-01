"""Kleiner HTTP-Client für den Juror-Scoring-Endpoint.

Bietet eine Klasse `JurorClient` mit Methoden um ein Bild per HTTP POST an
http://localhost:5010/scoring zu schicken.
"""
import base64
from typing import Any, Optional

import httpx

from juror_shared.ScoringRequestPayload import ScoringRequestPayload
from juror_shared.ScoringResponsePayload import ScoringResponsePayload


class JurorClient:
    """Client zum Senden von Bildern an den Juror-Server (/scoring).

    Beispiel:
        client = JurorClient()
        result = client.score_image("/path/to/image.jpg")
    """

    def __init__(self, base_url: str = "http://localhost:5010", timeout: int = 10):
        """Erzeuge einen neuen Client.

        Args:
            base_url: Basis-URL des Juror-Servers (Standard: http://localhost:5010)
            timeout: HTTP-Timeout in Sekunden für Requests
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def score_image(self, image_path: str, field_name: str = "image") -> ScoringResponsePayload | str:
        """Sende eine Bilddatei als Base64-JSON an /score_base64.

        Args:
            image_path: Pfad zur Bilddatei auf dem Dateisystem.
            field_name: (unused) früher für Multipart; bleibt zur API-Kompatibilität.

        Returns:
            Das vom Server zurückgegebene JSON (parsed) oder Text, falls kein JSON.

        Raises:
            FileNotFoundError: Wenn die Bilddatei nicht existiert.
            httpx.HTTPError: Bei HTTP/Netzwerkfehlern.
        """
        import os

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        url = f"{self.base_url}/score"

        with open(image_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode("ascii")
            payload = ScoringRequestPayload(filename=image_path, b64=b64)
            # model_dump() liefert ein dict, das httpx mit json= korrekt serialisiert
            json_payload = payload.model_dump()
            print("Sending image for scoring:", json_payload)
            resp = httpx.post(url, json=json_payload, timeout=self.timeout, headers={"Content-Type": "application/json"})

        # Raise for HTTP errors (4xx/5xx)
        resp.raise_for_status()

        # Versuche JSON zurückzugeben, fallback auf Text
        try:
            return ScoringResponsePayload(**resp.json());
        except ValueError:
            return resp.text

