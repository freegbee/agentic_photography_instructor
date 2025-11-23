"""In-memory LRU-Cache für `JurorService`.

Diese Datei enthält `JurorCachingService`, einen dekorator-artigen Wrapper, der
Anfragen an `score_ndarray` basierend auf einem inhaltsbasierten Hash cached.

Eigenschaften:
- LRU-Size-Limit (Anzahl Einträge)
- Optionale TTL (Sekunden)
- In-flight Deduplizierung: parallele Anfragen für denselben Key werden
  zusammengeführt, so dass nur eine Anfrage an den unterliegenden Service geht.

Alle Docstrings und Kommentare sind auf Deutsch.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from juror_client.juror_service import JurorService


def _hash_ndarray(array: np.ndarray) -> str:
    """Erzeuge einen stabilen, inhaltsbasierten Hash für ein numpy.ndarray.

    Eingeschlossene Eigenschaften:
    - Kontiguierliche Bytes der Daten (np.ascontiguousarray)
    - Dtype
    - Shape

    Rückgabe: Hexadezimaler SHA256-String.
    """
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    # shape and dtype deterministisch einschliessen
    meta = f"shape={arr.shape};dtype={str(arr.dtype)};"
    h.update(meta.encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


class JurorCachingService(JurorService):
    """Wrapper-Service mit in-memory LRU-Cache für `score_ndarray`.

    Args:
        inner: Die zu dekorierende `JurorService`-Instanz (z. B. JurorHttpService).
        maxsize: Maximale Anzahl Cache-Einträge (LRU).
        ttl: Optionale Time-to-live in Sekunden für Cache-Einträge. Wenn None,
             werden Einträge nicht zeitlich begrenzt.
        in_flight_grace: Sekunden, wie lange ein in-flight-Eintrag nach Abschluss
                         erhalten bleibt, damit wartende Threads das Ergebnis
                         lesen können.
    """

    def __init__(self, inner: JurorService, maxsize: int = 1024, ttl: Optional[float] = None,
                 in_flight_grace: float = 60.0) -> None:
        if inner is None:
            raise ValueError("inner (JurorService) darf nicht None sein")
        if maxsize <= 0:
            raise ValueError("maxsize muss > 0 sein")

        self._inner = inner
        self._maxsize = int(maxsize)
        self._ttl = None if ttl is None else float(ttl)
        self._in_flight_grace = float(in_flight_grace)

        # Cache: key -> (timestamp_seconds, value)
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()

        # In-flight: key -> {event: Event, exc: Optional[Exception]}
        self._in_flight: Dict[str, Dict[str, Any]] = {}

        self._lock = threading.Lock()

        # Metriken: einfache in-memory Zähler für Hits und Misses
        # Zugriff auf diese Zähler erfolgt unter self._lock, damit sie thread-safe sind.
        self._hits = 0
        self._misses = 0

    # --- Hilfsoperationen ---
    def _make_key(self, array: np.ndarray, filename: Optional[str], encoding: str) -> str:
        """Kombiniert Array-Hash mit Parametern zu einem eindeutigen Cache-Key."""
        arr_hash = _hash_ndarray(array)
        fname = filename or ""
        enc = encoding or "npy"
        key = f"{arr_hash}|fn={fname}|enc={enc}"
        return key

    def _prune_expired_locked(self) -> None:
        """Entfernt abgelaufene Einträge aus dem Cache. Muss unter Lock aufgerufen werden."""
        if self._ttl is None:
            return
        now = time.time()
        keys_to_delete = []
        for k, (ts, _) in self._cache.items():
            if now - ts > self._ttl:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            try:
                del self._cache[k]
            except KeyError:
                pass

    def _ensure_capacity_locked(self) -> None:
        """Stellt sicher, dass die Cache-Grösse das Limit nicht überschreitet (LRU)."""
        # Entferne nur, wenn die Anzahl Einträge das Limit überschreitet.
        while len(self._cache) > self._maxsize:
            # popitem(last=False) entfernt das älteste Element (LRU)
            try:
                self._cache.popitem(last=False)
            except Exception:
                break

    def _schedule_in_flight_cleanup(self, key: str) -> None:
        """Startet einen Timer, der den in-flight Eintrag nach einer Gnadenfrist löscht."""

        def _cleanup() -> None:
            with self._lock:
                self._in_flight.pop(key, None)

        t = threading.Timer(self._in_flight_grace, _cleanup)
        t.daemon = True
        t.start()

    # --- Öffentliche API ---
    def score_image(self, image_path: str):
        """Leitet Bild-Scoring an den inneren Service weiter (kein Caching)."""
        return self._inner.score_image(image_path)

    def score_ndarray(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy"):
        """Versucht, das Ergebnis aus dem Cache zu lesen; bei Fehlen delegiert an inner.

        Implementiert in-flight dedupe: parallele Anfragen für denselben Key
        warten auf das Ergebnis der ersten Anfrage.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array muss ein numpy.ndarray sein")

        key = self._make_key(array, filename, encoding)

        # 1) Schnellabfrage unter Lock: Cache oder in-flight prüfen
        with self._lock:
            # Abgelaufene entfernen
            self._prune_expired_locked()

            # Cache hit?
            if key in self._cache:
                ts, val = self._cache.pop(key)
                # Reinsert als jüngstes Element (LRU)
                self._cache[key] = (ts, val)
                # Cache-Hit zählen
                self._hits += 1
                return val

            # Wenn bereits eine Anfrage läuft: warten
            if key in self._in_flight:
                event = self._in_flight[key]["event"]
                # Wir warten ausserhalb der Lock
                await_other_call = True
            else:
                # Markiere als in-flight und wir führen die Anfrage aus
                event = threading.Event()
                self._in_flight[key] = {"event": event, "exc": None}
                await_other_call = False

        if await_other_call:
            # Warten auf die laufende Anfrage
            event.wait()
            # Nach dem Signal versuchen wir erneut, aus dem Cache zu lesen
            with self._lock:
                if key in self._cache:
                    ts, val = self._cache.pop(key)
                    self._cache[key] = (ts, val)
                    # Wartende, die nun das Ergebnis aus dem Cache lesen, sind Hits
                    self._hits += 1
                    return val
                # Falls kein Cached Result vorhanden ist, kann in_flight eine Exception gehalten haben
                in_f = self._in_flight.get(key)
                if in_f is not None and in_f.get("exc") is not None:
                    raise in_f["exc"]
            # Falls alles fehlschlägt, versuchen wir nochmals, die Anfrage selbst auszuführen
            # (Fällt selten an, z. B. wenn Producer einen Fehler warf)

        # Dieser Thread wird die eigentliche Anfrage ausführen -> Miss zählen
        with self._lock:
            self._misses += 1

        # Wir sind der ausführende Thread: führe die Anfrage ohne Lock aus
        exc = None
        result = None
        try:
            result = self._inner.score_ndarray(array=array, filename=filename, encoding=encoding)
        except Exception as e:
            exc = e
        finally:
            # Setze Ergebnis/Exception und wecke Wartende
            with self._lock:
                if exc is None and result is not None:
                    # Cache speichern mit Timestamp
                    self._cache[key] = (time.time(), result)
                    # LRU: move to end is automatic by reassigning
                    self._ensure_capacity_locked()
                # Trage Exception ein, falls vorhanden
                in_f = self._in_flight.get(key)
                if in_f is not None:
                    in_f["exc"] = exc
                    in_f["event"].set()
                    # Wenn ein Fehler aufgetreten ist, entfernen wir den in-flight
                    # Eintrag unmittelbar nach dem Signalisieren, damit nachfolgende
                    # Aufrufer einen neuen Versuch starten (statt sofort dieselbe
                    # Exception zu erhalten). Wartende Threads werden durch das
                    # Event geweckt und können entscheiden, ob sie erneut anfragen.
                    #
                    # Falls kein Fehler aufgetreten ist, behalten wir den in-flight
                    # Eintrag kurz und planen eine Aufräumung.
                    if exc is None:
                        self._schedule_in_flight_cleanup(key)
                    else:
                        # Entferne den in-flight-Eintrag sofort
                        try:
                            self._in_flight.pop(key, None)
                        except Exception:
                            pass

        if exc is not None:
            # Fehler an Aufrufer weiterreichen
            raise exc

        return result

    # --- Hilfs-APIs ---
    def clear_cache(self) -> None:
        """Leert den Cache (nützlich in Tests)."""
        with self._lock:
            self._cache.clear()

    def close(self) -> None:
        """Gibt Ressourcen frei und schliesst den inneren Service."""
        try:
            self._inner.close()
        except Exception:
            pass

    # --- Öffentliche Introspektions-/Manipulations-API ---
    def make_cache_key(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy") -> str:
        """Erzeuge den Cache-Key für ein Array (öffentlicher Wrapper für Tests/Introspektion).

        Rückgabe: String-Key, der zur Identifikation des Arrays im Cache verwendet wird.
        """
        return self._make_key(array, filename, encoding)

    def cached_keys(self) -> list:
        """Gibt thread-sicher die Liste der aktuellen Cache-Keys in LRU-Reihenfolge zurück.

        Wichtig: Dies ist eine Kopie; Veränderungen an der Rückgabe beeinflussen nicht den Cache.
        """
        with self._lock:
            return list(self._cache.keys())

    def contains(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy") -> bool:
        """Prüft, ob ein Array (bzw. sein Key) aktuell im Cache vorhanden ist.

        Berücksichtigt TTL (abgelaufene Einträge werden vor der Prüfung entfernt).
        """
        key = self._make_key(array, filename, encoding)
        with self._lock:
            self._prune_expired_locked()
            return key in self._cache

    def get_cached(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy"):
        """Liefert den gecachten Wert zurück oder `None`, wenn nicht vorhanden.

        Diese Funktion ist thread-safe und berücksichtigt TTL.
        """
        key = self._make_key(array, filename, encoding)
        with self._lock:
            self._prune_expired_locked()
            entry = self._cache.get(key)
            if entry is None:
                return None
            # Behalte LRU-Eigenschaft: reinsert
            ts, val = self._cache.pop(key)
            self._cache[key] = (ts, val)
            return val

    def invalidate_key(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy") -> None:
        """Entfernt explizit einen bestimmten Key aus dem Cache (falls vorhanden)."""
        key = self._make_key(array, filename, encoding)
        with self._lock:
            try:
                self._cache.pop(key, None)
            except Exception:
                pass

    def get_metrics(self) -> Dict[str, int]:
        """Gibt die aktuellen Cache-Metriken zurück (hits, misses, aktuelle Grösse).

        Rückgabe:
            Dict mit Schlüsseln 'hits', 'misses', 'size'.
        """
        with self._lock:
            return {"hits": int(self._hits), "misses": int(self._misses), "size": len(self._cache)}
