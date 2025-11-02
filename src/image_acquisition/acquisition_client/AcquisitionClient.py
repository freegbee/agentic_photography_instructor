import logging
from typing import Optional

import httpx

from image_acquisition.acquisition_shared.models_v1 import StartAsyncImageAcquisitionRequestV1, \
    AsyncImageAcquisitionJobResponseV1

logger = logging.getLogger(__name__)

class AcquisitionClient:
    """Client zum Starten von asynschtnen Image Acquisition jobs."""

    def __init__(
            self,
            # FIXME: Die URL ist hier noch hardcoded, sollte aber konfigurierbar sein
            base_url: str = "http://localhost:5005",
            timeout: int = 10,
            client: Optional[httpx.Client] = None,):
        """Erzeuge einen neuen Client.

        Args:
            base_url: Basis-URL des Acquisition-Servers (Standard: http://localhost:5000)
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

    def start_async_image_acquisition(self, dataset_id: str) -> str:
        """Starte einen asynchronen Image Acquisition Job für ein Dataset."""
        url = f"{self.base_url}/acquisition"
        logger.debug(f"Starte asynchronen image acquisition job für dataset {dataset_id} bei {url}")
        payload = StartAsyncImageAcquisitionRequestV1(**{"dataset_id": dataset_id})
        response = self._client.post(url, json=payload.model_dump(),  timeout=self.timeout, headers={"Content-Type": "application/json"})

        response.raise_for_status()

        try:
            response_object = AsyncImageAcquisitionJobResponseV1(**response.json())
            return response_object.job_uuid
        except ValueError:
            return response.text

    def query_async_image_acquisition_job(self, job_uui: str) -> Optional[AsyncImageAcquisitionJobResponseV1]:
        """Frage den Status eines asynchronen Image Acquisition Jobs ab."""
        url = f"{self.base_url}/acquisition/jobs/{job_uui}"
        logger.debug(f"Frage status des asynchronen image acquisition jobs {job_uui} bei {url}")
        response = self._client.get(url, timeout=self.timeout)

        if response.status_code == 404:
            return None

        response.raise_for_status()

        try:
            response_object = AsyncImageAcquisitionJobResponseV1(**response.json())
            return response_object
        except ValueError:
            logger.error(f"Fehler beim Parsen der Antwort des Acquisition-Servers: {response.text}")
            return None

    def close(self):
        """Schliesse den HTTP-Client, falls dieser intern erstellt wurde."""
        if not self._external_client:
            try:
                self._client.close()
            except Exception:
                logger.exception("Fehler beim Schliessen des HTTP-Clients")

    def __enter__(self) -> "AcquisitionClient":
        """Context-Manager entry: Wird verwendet im Rahmen des Python-Context-Manager-Protocol (aka with-Anweisung)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context-Manager exit: Wird beim Verlassen der with-Anweisung verwendet und schliesst den httpx client"""
        self.close()