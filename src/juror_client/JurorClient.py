"""Kleiner Client-Wrapper für den Juror-Scoring-Endpoint.

Diese Datei enthält die Klasse `JurorClient`, welche nun eine `JurorService`
Instanz nutzt, um die eigentliche Kommunikation durchzuführen. Dadurch ist die
HTTP-Implementierung (in `juror_service.JurorHttpService`) entkoppelt und kann
einfach in Tests gemockt oder ersetzt werden.
"""

import logging
from typing import Optional, Union, cast, Dict

import numpy as np

from juror_shared.models_v1 import ScoringResponsePayloadV1

from juror_client.juror_service import JurorService
from juror_client.registry import get_juror_service

logger = logging.getLogger(__name__)


class JurorClient:
    """Facade für Juror-Scoring-Funktionalität.

    `JurorClient` delegiert alle Operationen an eine `JurorService`-Instanz.
    Standardmässig wird `JurorHttpService` verwendet, um die bisherigen
    HTTP-Requests auszuführen. In Tests oder speziellen Umgebungen kann eine
    alternative Implementierung injiziert werden.

    Zusätzlich registriert der Konstruktor die verwendete `JurorService`-Instanz
    in der internen Registry unter dem Namen `register_name`, sofern dort noch
    kein Service vorhanden ist. Standardname ist "default_juror_client".
    """

    def __init__(
            self,
            # FIXME: Die URL ist hier noch hardcoded, sollte aber konfigurierbar sein
            base_url: str = "http://localhost:5010",
            timeout: float = 10.0,
            client: Optional[object] = None,
            service: Optional[JurorService] = None,
            register_name: str = "default_juror_client",
            use_cache: bool = True,
            cache_maxsize: int = 1024,
            cache_ttl: Optional[float] = None,
    ):
        """Erzeuge einen neuen `JurorClient`.

        Args:
            base_url: Basis-URL des Juror-Servers (Standard: http://localhost:5010)
            timeout: HTTP-Timeout in Sekunden für Requests
            client: Optionaler HTTP-Client, wird an die `JurorHttpService` weitergegeben
            service: Optional: eine konkrete `JurorService`-Instanz. Falls angegeben,
                     wird diese anstelle eines `JurorHttpService` verwendet.
            register_name: Optionaler Registry-Name; wenn angegeben (Standard
                           "default_juror_client"), wird die verwendete Service-
                           Instanz unter diesem Namen registriert, falls dort
                           noch keine Instanz existiert.
        """

        # Verwende die zentrale Fabrik/Registry, damit Caching-Wrapper und
        # registrierte Instanzen korrekt gehandhabt werden. get_juror_service
        # übernimmt die Logik: falls `service` übergeben ist, wird diese Instanz
        # verwendet; falls `register_name` angegeben und bereits registriert,
        # wird die registrierte Instanz zurückgegeben; sonst wird eine neue
        # HttpService erzeugt und optional mit JurorCachingService umhüllt.
        self._service = None
        try:
            self._service = get_juror_service(
                name=register_name,
                service=service,
                base_url=base_url,
                timeout=timeout,
                client=cast(Optional[object], client),
                use_cache=use_cache,
                cache_maxsize=cache_maxsize,
                cache_ttl=cache_ttl,
            )
        except Exception:
            # Fallback: falls die Fabrik versagt, werfe eine aussagekräftige Exception
            logger.exception("Failed to obtain JurorService via get_juror_service; falling back to direct HTTP service")
            # create direct HTTP service as last resort (no caching)
            from juror_client.juror_service import JurorHttpService
            self._service = JurorHttpService(base_url=base_url, timeout=timeout, client=cast(Optional[object], client))

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

    def get_cache_metrics(self) -> Dict[str, int]:
        """Gebe Cache-Metriken des zugrundeliegenden Juror-Services zurück.

        Falls der aktuelle Service einen Cache unterstützt und eine
        `get_metrics()`-Methode anbietet, werden die Felder `hits`, `misses` und
        `size` ausgelesen und als ints zurückgegeben. Andernfalls wird ein
        Default-Dict mit 0-Werten zurückgegeben.
        """
        try:
            if self._service is None:
                return {"hits": 0, "misses": 0, "size": 0}
            else:
                metrics = self._service.get_metrics()
                return {
                    "hits": int(metrics.get('hits', 0)),
                    "misses": int(metrics.get('misses', 0)),
                    "size": int(metrics.get('size', 0)),
                }
        except Exception:
            logger.exception("Failed to read cache metrics from underlying juror service")
        return {"hits": 0, "misses": 0, "size": 0}
