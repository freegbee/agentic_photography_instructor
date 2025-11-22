"""Kleiner Client-Wrapper für den Juror-Scoring-Endpoint.

Diese Datei enthält die Klasse `JurorClient`, welche nun eine `JurorService`
Instanz nutzt, um die eigentliche Kommunikation durchzuführen. Dadurch ist die
HTTP-Implementierung (in `juror_service.JurorHttpService`) entkoppelt und kann
einfach in Tests gemockt oder ersetzt werden.
"""

import logging
from typing import Optional, Union, cast

import numpy as np

from juror_shared.models_v1 import ScoringResponsePayloadV1

from juror_client.juror_service import JurorService, JurorHttpService

logger = logging.getLogger(__name__)


class JurorClient:
    """Facade für Juror-Scoring-Funktionalität.

    `JurorClient` delegiert alle Operationen an eine `JurorService`-Instanz.
    Standardmässig wird `JurorHttpService` verwendet, um die bisherigen
    HTTP-Requests auszuführen. In Tests oder speziellen Umgebungen kann eine
    alternative Implementierung injiziert werden.
    """

    def __init__(
            self,
            # FIXME: Die URL ist hier noch hardcoded, sollte aber konfigurierbar sein
            base_url: str = "http://localhost:5010",
            timeout: float = 10.0,
            client: Optional[object] = None,
            service: Optional[JurorService] = None,
    ):
        """Erzeuge einen neuen `JurorClient`.

        Args:
            base_url: Basis-URL des Juror-Servers (Standard: http://localhost:5010)
            timeout: HTTP-Timeout in Sekunden für Requests
            client: Optionaler HTTP-Client, wird an die `JurorHttpService` weitergegeben
            service: Optional: eine konkrete `JurorService`-Instanz. Falls angegeben,
                     wird diese anstelle eines `JurorHttpService` verwendet.
        """

        # Falls bereits ein Service übergeben wurde, diesen verwenden.
        if service is not None:
            if not isinstance(service, JurorService):
                raise TypeError("service muss eine Instanz von JurorService sein")
            self._service = service
        else:
            # Sonst eine Standard-HTTP-Service-Instanz erstellen
            self._service = JurorHttpService(base_url=base_url, timeout=timeout, client=cast(Optional[object], client))  # type: ignore[arg-type]

    def score_image(self, image_path: str) -> Union[ScoringResponsePayloadV1, str]:
        """Delegiert an `JurorService.score_image`.

        Diese Methode dient der Rückwärtskompatibilität mit dem bisherigen API.
        """
        return self._service.score_image(image_path)

    def score_ndarray(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy") -> Union[ScoringResponsePayloadV1, str]:
        """Delegiert an `JurorService.score_ndarray`.

        Das Array wird an den darunterliegenden Service weitergereicht. Die
        Implementierung (HTTP, Mock, Cache, ...) entscheidet, wie das Array
        serialisiert und gesendet wird.
        """
        return self._service.score_ndarray(array=array, filename=filename, encoding=encoding)

    def close(self) -> None:
        """Gibt Ressourcen des zugrundeliegenden Service frei (falls vorhanden)."""
        try:
            self._service.close()
        except Exception:
            logger.exception("Fehler beim Schliessen des JurorService")

    def __enter__(self) -> "JurorClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
