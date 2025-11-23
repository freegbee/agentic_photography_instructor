import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, Set

from utils.ImageUtils import ImageUtils

logger = logging.getLogger(__name__)


class AsyncImageSaver:
    """Asynchroner, begrenzter ThreadPool für Image-Write-Operationen.

    Eigenschaften zur Vermeidung von Memory-Growth bei sehr großen Batches:
    - begrenzte Anzahl Worker-Threads (max_workers)
    - Semaphore begrenzt die Anzahl paralleler/enqueued Schreib-Tasks (max_queue_size)
    - futures werden bei Abschluss aus der internen Menge entfernt, um Referenzen freizugeben
    """
    def __init__(self, max_workers: int = 4, max_queue_size: int = 256):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = threading.Semaphore(max_queue_size)
        self._futures: Set = set()
        self._lock = threading.Lock()

    def save_async(self, img_ndarray: Any, path: str):
        # Acquire semaphore to apply backpressure when too many pending writes exist
        self._semaphore.acquire()
        try:
            fut = self._executor.submit(self._save, img_ndarray, path)
        except Exception:
            # Ensure semaphore released on submit failure
            self._semaphore.release()
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

        fut.add_done_callback(_on_done)
        return fut

    @staticmethod
    def _save(img, path: str) -> bool:
        # Delegiere das tatsächliche Schreiben an die vorhandene Utility
        ImageUtils.save_image(img, path)
        return True

    def shutdown(self, wait: bool = True):
        # Warte auf alle noch laufenden futures
        if wait:
            with self._lock:
                futures = list(self._futures)
            if futures:
                from concurrent.futures import wait as _cf_wait
                _cf_wait(futures)
        self._executor.shutdown(wait=wait)
