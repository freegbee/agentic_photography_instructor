import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent.futures as _cf
import threading
from typing import Any, Set

from utils.ImageUtils import ImageUtils

logger = logging.getLogger(__name__)


def _save_image_task(img, path: str) -> bool:
    """Top-level save function used by worker processes (picklable)."""
    ImageUtils.save_image(img, path)
    return True


class AsyncImageSaver:
    """Asynchroner, begrenzter Process/ThreadPool für Image-Write-Operationen.

    Unterstützt drei executor_type values:
    - 'process' (default): ProcessPoolExecutor (gute CPU-Parallelität, pickling der Arrays)
    - 'thread': ThreadPoolExecutor (keine Pickle-Kopien, sinnvoll für I/O-bound)
    - 'noop': kein echtes Schreiben, Future wird sofort als erfolgreich abgeschlossen (zur Messung)

    Eigenschaften zur Vermeidung von Memory-Growth bei sehr großen Batches:
    - begrenzte Anzahl Worker-Prozesse/Threads (max_workers)
    - Semaphore begrenzt die Anzahl paralleler/enqueued Schreib-Tasks (max_queue_size)
    - futures werden bei Abschluss aus der internen Menge entfernt, um Referenzen freizugeben
    """
    def __init__(self, max_workers: int = 4, max_queue_size: int = 256, executor_type: str = "process"):
        """executor_type: 'process' or 'thread' or 'noop'.
        'process' uses ProcessPoolExecutor (default),
        'thread' uses ThreadPoolExecutor,
        'noop' performs no write and completes immediately.
        """
        if executor_type not in ("process", "thread", "noop"):
            raise ValueError("executor_type must be 'process', 'thread' or 'noop'")
        self._executor_type = executor_type
        if executor_type == "process":
            self._executor = ProcessPoolExecutor(max_workers=max_workers)
        elif executor_type == "thread":
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self._executor = None
        self._semaphore = threading.Semaphore(max_queue_size)
        self._futures: Set = set()
        self._lock = threading.Lock()

    def save_async(self, img_ndarray: Any, path: str):
        # Acquire semaphore to apply backpressure when too many pending writes exist
        self._semaphore.acquire()
        try:
            if self._executor_type == "noop":
                # create a Future that is already completed successfully
                fut = _cf.Future()
                fut.set_result(True)
            else:
                # For process executor we must submit picklable top-level function; for thread, same function works
                fut = self._executor.submit(_save_image_task, img_ndarray, path)
        except Exception:
            # Ensure semaphore released on submit failure
            try:
                self._semaphore.release()
            except Exception:
                pass
            raise

        with self._lock:
            self._futures.add(fut)

        def _on_done(f):
            try:
                # propagate exceptions to log
                f.result()
            except Exception:
                logger.exception("Async save failed for %s", path)
            finally:
                with self._lock:
                    self._futures.discard(f)
                # release slot
                try:
                    self._semaphore.release()
                except Exception:
                    pass

        # If future is already done, add_done_callback will call immediately
        fut.add_done_callback(_on_done)
        return fut

    def shutdown(self, wait: bool = True):
        # Warte auf alle noch laufenden futures
        if wait:
            with self._lock:
                futures = list(self._futures)
            if futures:
                from concurrent.futures import wait as _cf_wait
                _cf_wait(futures)
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
