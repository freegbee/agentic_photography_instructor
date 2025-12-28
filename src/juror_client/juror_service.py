"""Dienst-Interface und HTTP-Implementierung für den Juror-Scoring-Service.

Dieses Modul definiert:
- `JurorService` (Abstraktes Interface) mit den benötigten Methoden.
- `JurorHttpService` als konkrete Implementierung, die die bisherigen HTTP-Aufrufe
  aus `JurorClient` kapselt.

Docstrings und Kommentare sind auf Deutsch gehalten.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Union, BinaryIO, cast

import httpx
import numpy as np

from juror_shared.models_v1 import ScoringRequestPayloadV1, ScoringResponsePayloadV1

logger = logging.getLogger(__name__)


class JurorService(ABC):
    """Abstraktes Interface für Juror-Scoring-Dienste.

    Konkrete Implementierungen müssen mindestens die Methoden
    `score_image` und `score_ndarray` bereitstellen. Optional kann `close`
    überschrieben werden, um Ressourcen (z. B. HTTP-Clients) freizugeben.
    """

    @abstractmethod
    def score_image(self, image_path: str) -> Union[ScoringResponsePayloadV1, str]:
        """Scoring für eine Bilddatei ausführen.

        Args:
            image_path: Pfad der Bilddatei auf dem Filesystem.

        Returns:
            Ein `ScoringResponsePayloadV1`-Objekt oder ein Text-Fallback.
        """
        raise NotImplementedError

    @abstractmethod
    def score_ndarray(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy") -> Union[ScoringResponsePayloadV1, str]:
        """Scoring für ein numpy.ndarray ausführen.

        Args:
            array: Das zu sendende numpy.ndarray.
            filename: Optionaler Dateiname für die Multipart-Datei.
            encoding: 'npy' oder 'npz'.

        Returns:
            Ein `ScoringResponsePayloadV1`-Objekt oder ein Text-Fallback.
        """
        raise NotImplementedError

    def get_metrics(self) -> dict:
        """Optional: Gebe Metriken des Services zurück (z. B. Cache-Statistiken).

        Returns:
            Ein Dictionary mit Metriken.
        """
        return {}

    def close(self) -> None:  # pragma: no cover - trivial default
        """Standard-Implementierung: keine Aktion. Konkrete Klassen können
        Ressourcen freigeben.
        """
        return None


class JurorHttpService(JurorService):
    """Konkrete Implementierung, die HTTP-Requests an den Juror-Server sendet.

    Die Implementierung ist bewusst sehr nah an der bisherigen `JurorClient`
    Implementierung, jedoch in eine eigene Klasse ausgelagert, so dass sie
    einfach in Tests gemockt oder durch andere Implementierungen ersetzt
    werden kann.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5010",
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        # Validierungen
        if base_url is None:
            raise ValueError("base_url darf nicht None sein")
        if not isinstance(base_url, str):
            raise TypeError("base_url muss vom Typ str sein")

        self._api_version = "v1"
        self.base_url = base_url.rstrip("/") + f"/{self._api_version}/"
        self.scoring_endpoint = "score"

        self.timeout = float(timeout)

        self._external_client = client is not None
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        if client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, limits=limits)
        else:
            self._client = client

    def score_image(self, image_path: str) -> Union[ScoringResponsePayloadV1, str]:
        """Sende eine Bilddatei per Base64-JSON an /score.

        Funktionalität entspricht der bisherigen Implementation im Projekt.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.debug("Posting file %s to %s", image_path, self.base_url + self.scoring_endpoint)
        with open(image_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode("ascii")
            payload = ScoringRequestPayloadV1(filename=os.path.basename(image_path), b64=b64)
            json_payload = payload.model_dump()
            logger.debug("Sending image for scoring: %s", json_payload)
            resp = self._client.post(self.scoring_endpoint, json=json_payload)

        resp.raise_for_status()

        try:
            return ScoringResponsePayloadV1(**resp.json())
        except ValueError:
            return resp.text

    def score_ndarray(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy") -> Union[ScoringResponsePayloadV1, str]:
        """Sende ein numpy.ndarray via multipart/form-data.

        Das Array wird als .npy oder .npz in den Request-Body geschrieben.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array muss ein numpy.ndarray sein")

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
        """Schliesse den internen HTTP-Client, falls er intern erstellt wurde."""
        if not getattr(self, "_external_client", False):
            try:
                self._client.close()
            except Exception:
                logger.exception("Fehler beim Schliessen des HTTP-Clients")

