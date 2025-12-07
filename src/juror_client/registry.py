"""Factory und einfache Service-Registry für `JurorService`-Instanzen.

Dieses Modul stellt folgende Utilities bereit:
- `get_juror_service(...)`: Fabrikfunktion, die eine konfigurierte `JurorService`
  Instanz erzeugt (z. B. `JurorHttpService`) und optional mit
  `JurorCachingService` umhüllt.
- Eine minimale Registry (`register_service`, `get_registered_service`, `clear_registry`)
  um einmal erzeugte Instanzen wiederzuverwenden.

Beispiele zur Nutzung:
- service = get_juror_service() # Neuen Service erstellen
- service = get_juror_service(use_cache=True, cache_maxsize=512, cache_ttl=3600) # Neuer Service mit Cache
- service = get_juror_service(name="default", use_cache=True) # Registrierten oder neuen Service mit Cache und gegebenem Namen
- service2 = get_juror_service(name="default") # returns same instance # Selbe Instanz des Services wiederverwenden

Nutzung für Tests:
1. get_juror_service(service=mock_service) # Mock-Service übergeben
2. clear_registry() # am Ende des Tests aufrufen, um Seiteneffekte zu vermeiden.


"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import httpx

from juror_client.juror_cache import JurorCachingService
from juror_client.juror_service import JurorHttpService, JurorService
from juror_client.local_juror_service import LocalJurorService

logger = logging.getLogger(__name__)

# Minimale globale Registry: name -> JurorService
_SERVICE_REGISTRY: Dict[str, JurorService] = {}


def register_service(name: str, service: JurorService) -> None:
    """Registriert eine `JurorService`-Instanz unter einem Namen.

    Nützlich, wenn mehrere Komponenten dieselbe Instanz wiederverwenden sollen.
    """
    if not name:
        raise ValueError("name darf nicht leer sein")
    if service is None:
        raise ValueError("service darf nicht None sein")
    _SERVICE_REGISTRY[name] = service
    logger.debug("JurorService unter Namen '%s' registriert", name)


def get_registered_service(name: str) -> Optional[JurorService]:
    """Gibt die registrierte Service-Instanz zurück oder None.

    Achtung: Gibt eine Referenz auf die registrierte Instanz zurück (keine Kopie).
    """
    return _SERVICE_REGISTRY.get(name)


def clear_registry() -> None:
    """Leert die interne Service-Registry (vor allem für Tests nützlich)."""
    _SERVICE_REGISTRY.clear()


def get_juror_service(
        *,
        name: Optional[str] = None,
        service: Optional[JurorService] = None,
        base_url: str = "http://localhost:5010",
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
        use_local: bool = False,
        use_cache: bool = False,
        cache_maxsize: int = 1024,
        cache_ttl: Optional[float] = None,
) -> JurorService:
    """Fabrikfunktion, die eine `JurorService` liefert.

    Aufruf-Logik (vereinfachte Priorität):
    1. Wenn `service` angegeben ist, wird diese Instanz direkt zurückgegeben.
    2. Falls `name` angegeben und bereits registriert, wird die registrierte
       Instanz zurückgegeben.
    3. Sonst wird eine neue `JurorHttpService` erstellt (mit `base_url`, `timeout`).
       Falls `use_cache=True` wird das Resultat mit `JurorCachingService` umhüllt.

    Wenn `name` angegeben ist, registriert die Factory die erzeugte Instanz
    automatisch unter diesem Namen, damit spätere Aufrufe dieselbe Instanz
    zurückgeben können.

    Args:
        name: Optionaler Registry-Name. Wenn angegeben, wird die Service-Instanz
              unter diesem Namen registriert/abgerufen.
        service: Optional: bereits erzeugte `JurorService`-Instanz (wird dann
                 direkt verwendet, Registry hat dann Vorrang vor Neuerzeugung).
                 Wird auch name angegeben, wird die Instanz zusätzlich registriert.
        base_url: Basis-URL für den HTTP-Service (falls dieser erzeugt wird).
        timeout: Timeout für HTTP-Client.
        client: Optionaler HTTP-Client, wird an `JurorHttpService` weitergereicht.
        use_local: JurorService lokal (ohne HTTP) verwenden.
        use_cache: Ob ein `JurorCachingService` um die erzeugte Service-Instanz
                   gelegt werden soll.
        cache_maxsize: Max-Anzahl Einträge im LRU-Cache.
        cache_ttl: Optional TTL für Cache-Einträge (Sekunden).

    Returns:
        Eine `JurorService`-Instanz (entweder übergeben, registriert oder neu erzeugt).
    """
    # 1) Wenn explizit eine Instanz übergeben wurde, nutze diese
    if service is not None:
        # Falls ein Name übergeben wurde, registriere die gegebene Instanz
        if name is not None:
            register_service(name, service)
        logger.debug("Returning provided JurorService instance")
        return service

    # 2) Wenn ein Name angegeben wurde und bereits registriert ist, liefere diese
    if name is not None:
        existing = get_registered_service(name)
        if existing is not None:
            logger.debug("Returning registered JurorService for name=%s", name)
            return existing

    final_service: JurorService
    if use_local:
        # 3) Erzeuge eine neue Http-Service-Instanz
        local_service = LocalJurorService()
        final_service: JurorService = local_service
    else:
        # 3) Erzeuge eine neue Http-Service-Instanz
        http_service = JurorHttpService(base_url=base_url, timeout=timeout, client=client)
        final_service: JurorService = http_service

    # Optional: Caching-Wrapper
    if use_cache:
        final_service = JurorCachingService(inner=final_service, maxsize=cache_maxsize, ttl=cache_ttl)

    # Falls ein Name angegeben wurde, registriere die erzeugte Instanz
    if name is not None:
        register_service(name, final_service)

    return final_service
