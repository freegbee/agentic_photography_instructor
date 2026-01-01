import logging
import multiprocessing
import queue
import time
from typing import Optional, Union, List
from multiprocessing import shared_memory

import numpy as np

from juror_client.juror_service import JurorService
from juror_server.juror.Juror import Juror
from juror_shared.models_v1 import ScoringResponsePayloadV1

logger = logging.getLogger(__name__)

def _juror_worker_process(request_queue: multiprocessing.Queue, worker_id: int):
    """
    Der Code, der im separaten GPU-Worker-Prozess läuft.
    """
    # Imports hier drin, damit CUDA nicht im Hauptprozess initialisiert wird (wichtig für 'spawn')
    import torch
    import cv2

    # Threading limitieren
    cv2.setNumThreads(0)
    torch.set_num_threads(1)

    logger.info(f"Worker {worker_id}: Loading Juror Model on GPU...")
    try:
        juror = Juror()
        logger.info(f"Worker {worker_id}: Model loaded. Ready for inference.")
    except Exception as e:
        logger.exception(f"Worker {worker_id}: Failed to load model.")
        return

    while True:
        try:
            # Warten auf Arbeit: (image_array, reply_queue)
            # NEU: Wir erwarten nun Metadaten für Shared Memory statt des Bildes selbst
            item = request_queue.get()
            if item is None:
                # Poison Pill empfangen -> Beenden
                break

            shm_name, shape, dtype, reply_queue = item

            # 1. Shared Memory öffnen
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            # 2. Array aus Shared Memory rekonstruieren (Zero-Copy View)
            image_rgb = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

            # Inference
            score = juror.inference(image_rgb)

            # 3. Shared Memory schließen (nicht unlinken, das macht der Sender!)
            existing_shm.close()

            # Ergebnis zurücksenden
            # Wir senden ein einfaches Dict oder Payload Objekt zurück
            result = ScoringResponsePayloadV1(score=float(score))
            reply_queue.put(result)

        except Exception as e:
            logger.error(f"Worker {worker_id}: Error during inference: {e}")
            # Versuchen, dem Aufrufer einen Fehler zu senden, falls möglich
            # (Hier vereinfacht weggelassen, Timeout beim Client fängt das ab)

    logger.info(f"Worker {worker_id}: Shutting down.")


class JurorQueueService(JurorService):
    """
    Ein Client-seitiger Service, der Anfragen in die globale Queue legt
    und auf die Antwort wartet. Wird in den Environments instanziiert.
    """
    def __init__(self, request_queue, reply_queue):
        self.request_queue = request_queue
        self.reply_queue = reply_queue

    def score_ndarray(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy") -> Union[ScoringResponsePayloadV1, str]:
        # Performance-Optimierung: Shared Memory nutzen
        # Statt das Bild durch die Queue zu quetschen (langsam bei Manager.Queue),
        # schreiben wir es in Shared Memory und senden nur den Pointer.
        
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        
        try:
            # Daten in Shared Memory kopieren
            shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
            shm_array[:] = array[:]

            # 2. Anfrage in die globale Worker-Queue legen (nur Metadaten!)
            self.request_queue.put((shm.name, array.shape, array.dtype, self.reply_queue))

            # 3. Auf Antwort warten (blockierend)
            result = self.reply_queue.get(timeout=30.0)
            return result

        except queue.Empty:
            logger.error("Timeout waiting for Juror Worker response.")
            return ScoringResponsePayloadV1(score=0.0)
        finally:
            # Aufräumen: Shared Memory schließen und freigeben
            shm.close()
            shm.unlink()

    def score_image(self, image_path: str) -> Union[ScoringResponsePayloadV1, str]:
        raise NotImplementedError("QueueService supports only ndarray currently for performance reasons.")

    def close(self):
        pass

    def get_metrics(self):
        return {}


class JurorWorkerPool:
    """
    Verwaltet die GPU-Prozesse und die Kommunikations-Queues.
    Lebt im Hauptprozess (Trainer).
    """
    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        # WICHTIG: Wir nutzen Manager.Queue, um den "RuntimeError" auf Windows/Mac zu vermeiden.
        # Dank SharedMemory ist die Performance trotzdem hoch.
        self.manager = multiprocessing.Manager()
        self.request_queue = self.manager.Queue()
        self.workers: List[multiprocessing.Process] = []

    def start(self):
        logger.info(f"Starting JurorWorkerPool with {self.num_workers} workers...")
        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=_juror_worker_process,
                args=(self.request_queue, i),
                daemon=True
            )
            p.start()
            self.workers.append(p)

    def stop(self):
        logger.info("Stopping JurorWorkerPool...")
        for _ in self.workers:
            self.request_queue.put(None) # Poison Pill
        for p in self.workers:
            p.join(timeout=5)
        self.manager.shutdown()