import json
import logging
from typing import Optional, Union

import httpx

from image_acquisition.acquisition_shared.models_v1 import StartAsyncImageAcquisitionRequestV1, \
    AsyncImageAcquisitionJobResponseV1

logger = logging.getLogger(__name__)

class AcquisitionClient:
    """Client zum Starten von asynchronen Image Acquisition jobs."""

    def __init__(
            self,
            base_url: Union[str, None] = None,
            timeout: float = 10.0,
            client: Optional[httpx.Client] = None,):
        """Erzeuge einen neuen Client.

        Args:
            base_url: Basis-URL des Acquisition-Servers (Standard: http://localhost:5005)
            timeout: HTTP-Timeout in Sekunden für Requests
            client: Client zum Starten von asynchronen Image Acquisition
        """

        # Validierungen der Parameter
        if base_url is None:
            raise ValueError("base_url darf nicht None sein. Beispiel: 'http://localhost:5005' - Abhängig den der lokalen Konfiguration oder Umgebungsvariable ab.")
        if not isinstance(base_url, str):
            raise TypeError("base_url muss vom Typ str sein")

        self._api_version = "v1"
        self.base_url = base_url.rstrip("/") + f"/{self._api_version}/" # trailing slash preserves path as directory
        self.acquisition_root_endpoint = "acquisition"
        self.acquisition_jobs_endpoint = f"{self.acquisition_root_endpoint}/jobs"

        self.timeout = float(timeout)

        # Connection pooling mit einem persistent httpx.Client
        # Falls ein externer Client übergeben wurde, diesen verwenden (und dann auch nicht schliessen im closer / im __exit__)
        self._external_client = client is not None
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        if client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, limits=limits, http2=True)
        else:
            self._client = client

    def start_async_image_acquisition(self, dataset_id: str) -> str:
        """Starte einen asynchronen Image-Acquisition-Job. Gibt die Job‑UUID zurück."""
        logger.debug("Starte asynchronen image acquisition job für dataset %s bei %s", dataset_id, self.base_url + self.acquisition_root_endpoint)
        payload = StartAsyncImageAcquisitionRequestV1(**{"dataset_id": dataset_id})
        response = self._client.post(self.acquisition_root_endpoint, json=payload.model_dump())

        response.raise_for_status()

        try:
            resp_json = response.json()
            response_object = AsyncImageAcquisitionJobResponseV1(**resp_json)
            return response_object.job_uuid
        except json.JSONDecodeError:
            logger.error("Antwort des Acquisition-Servers ist kein JSON: %s", response.text)
            return response.text

    def query_async_image_acquisition_job(self, job_uuid: str) -> Optional[AsyncImageAcquisitionJobResponseV1]:
        """Frage den Status eines asynchronen Image Acquisition Jobs ab."""
        endpoint = f"{self.acquisition_jobs_endpoint}/{job_uuid}"
        logger.debug("Frage status des asynchronen image acquisition jobs %s bei url %s", job_uuid, self.base_url + endpoint)

        response = self._client.get(endpoint, timeout=self.timeout)

        if response.status_code == 404:
            return None

        response.raise_for_status()

        try:
            response_object = AsyncImageAcquisitionJobResponseV1(**response.json())
            return response_object
        except json.JSONDecodeError:
            logger.error("Fehler beim Parsen der Antwort des Acquisition-Servers: %s", response.text)
            return None

    def close(self):
        """Schliesse den HTTP-Client, falls dieser intern erstellt wurde."""
        if not self._external_client:
            try:
                self._client.close()
            except Exception:
                logger.exception("Fehler beim Schliessen des HTTP-Clients")

    def __enter__(self) -> "AcquisitionClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()