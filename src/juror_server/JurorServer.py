import asyncio
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Response, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, process_collector, gc_collector, \
    platform_collector

# Import the Juror implementation from the same package
from juror.Juror import Juror
from juror_server.prometheus.Metrics import init_metrics
from juror_shared.models_v1 import ScoringRequestPayloadV1, ScoringResponsePayloadV1
from utils.LoggingUtils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Custom registry für Prometheus Metriken
registry = CollectorRegistry()
# Default collectors registrieren
gc_collector.GCCollector(registry=registry)
platform_collector.PlatformCollector(registry=registry)
process_collector.ProcessCollector(registry=registry)

# Custom metrics initialisieren
prometheus_metrics = init_metrics(registry=registry)

# Global juror-service instance (initialized in lifespan)
_juror: Optional[Juror] = None

# Hilfsfunktionen (synchron, werden mit to_thread aufgerufen)
def _decode_base64_to_array(payload: ScoringRequestPayloadV1) -> np.ndarray:
    data = base64.b64decode(payload.b64)
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(pil_img).astype(np.uint8)

def _load_numpy_bytes_to_array(data: bytes) -> np.ndarray:
    return load_numpy_from_bytes(data)  # bestehende Funktion, synchron

def _run_inference(arr: np.ndarray) -> float:
    # ggf. hier Locks nutzen, falls das Modell nicht thread-safe ist
    return float(_juror.inference(arr))


def load_numpy_from_bytes(data: bytes) -> np.ndarray:
    """Lädt ein `npy` oder `npz` aus Bytes und gibt ein ndarray zurück."""
    bio = io.BytesIO(data)
    loaded = np.load(bio, allow_pickle=False)
    try:
        # falls ein NpzFile (Archiv) vorliegt, nimm das erste Array
        if isinstance(loaded, np.lib.npyio.NpzFile):
            keys = list(loaded.files)
            if not keys:
                raise ValueError("npz archive contains no arrays")
            arr = loaded[keys[0]]
        else:
            # direktes ndarray (.npy)
            arr = loaded
        return arr.astype(np.uint8)
    finally:
        # NpzFile braucht explicit close(), ndarray nicht
        try:
            loaded.close()
        except Exception as e:
            logger.debug("Failed to close npz archive", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to initialize and cleanup the Juror instance."""
    global _juror
    if _juror is None:
        # instantiate the juror-service (this may load a model and take time)
        logger.info("Loading Juror model...")
        _juror = Juror()
        logger.info("Juror model loaded...")
        try:
            prometheus_metrics.JUROR_LOADED.set(1)
        except Exception as e:
            logger.exception(e)
    try:
        yield
    finally:
        _juror = None
        try:
            prometheus_metrics.JUROR_LOADED.set(0)
        except Exception as e:
            logger.exception(e)


app = FastAPI(title="Juror Inference API", lifespan=lifespan)

v1 = APIRouter(prefix="/v1")

# Allow all origins for convenience (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware: läuft vor und nach jedem Request; inkrementiert Request-Counter nach Antwort."""
    if request.url.path in ("/metrics", "/health"):
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)  # führt die Route aus und liefert die Response zurück
    elapsed = time.perf_counter() - start
    logger.info("Request to %s took %.3f seconds", request.url.path, elapsed)

    # Metriken für bestimmte Endpunkte überspringen
    try:
        prometheus_metrics.HTTP_REQUEST_DURATION_HIST.labels(
            status=str(response.status_code),
            endpoint=request.url.path,
            method=request.method
        ).observe(elapsed)
    except Exception as e:
        # Fehler beim Metrik-Update dürfen Request nicht brechen
        logger.exception(e)

    return response


@app.get("/health")
async def health():
    return {"status": "ok", "juror_loaded": _juror is not None}


@v1.post("/score", response_model=ScoringResponsePayloadV1)
async def score_image_base64(payload: ScoringRequestPayloadV1):
    try:
        data = base64.b64decode(payload.b64)
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        image_np = np.array(pil_img).astype(np.uint8)
    except Exception as e:
        prometheus_metrics.ERROR_COUNT.labels(type="decode_error").inc()
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {e}")

    if _juror is None:
        prometheus_metrics.ERROR_COUNT.labels(type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Juror model not loaded yet")

    try:
        start = time.perf_counter()
        score = float(_juror.inference(image_np))
        elapsed = time.perf_counter() - start

        # Metric für Inferenzzeit und erfolgreiche Anfrage
        try:
            prometheus_metrics.SCORING_DURATION_HIST.observe(elapsed)
            prometheus_metrics.SCORING_DURATION.set(elapsed)
            prometheus_metrics.SCORING_VALUE.set(score)
        except Exception as e:
            logger.exception(e)

        logger.info("Score: %.4f took %.2f seconds for json base64 upload, size %d bytes", score, elapsed, len(data))
    except Exception as e:
        prometheus_metrics.ERROR_COUNT.labels(type="inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    response = ScoringResponsePayloadV1(
        **{"score": score,
           "filename": payload.filename,
           "message": "Scoring successful"})
    return response


@v1.post("/score/ndarray", response_model=ScoringResponsePayloadV1)
async def score_ndarray_file(array_file: Optional[UploadFile] = File(None)):
    """Akzeptiert ein npy/npz File (Multipart Upload) und wertet es aus.

    Der Body sollte ein Multipart-Formular sein mit Feld `array_file` (binär)
    """
    if array_file is None:
        logger.warning("no array_file uploaded in request")
        prometheus_metrics.ERROR_COUNT.labels(type="no_file").inc()
        raise HTTPException(status_code=400, detail="No array_file uploaded")

    try:
        data = await array_file.read()
        filename_from = array_file.filename
        await array_file.close()
    except Exception as e:
        logger.warning("exception while reading uploaded file: %s", e)
        prometheus_metrics.ERROR_COUNT.labels(type="read_error").inc()
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded data: {e}")

    logger.debug("success reading data of length %d bytes", len(data))

    # Load numpy array from bytes (support npy and npz)
    try:
        image_np = await asyncio.to_thread(_load_numpy_bytes_to_array, data)
        logger.debug("image_np shape: %s, dtype: %s", image_np.shape, image_np.dtype)
    except Exception as e:
        logger.warning("exception decoding data: %s", e)
        prometheus_metrics.ERROR_COUNT.labels(type="decode_error").inc()
        raise HTTPException(status_code=400, detail=f"Failed to decode numpy array: {e}")

    if _juror is None:
        prometheus_metrics.ERROR_COUNT.labels(type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Juror model not loaded yet")

    try:
        logger.debug("start to score ndarray")
        start = time.perf_counter()
        score = await asyncio.to_thread(_run_inference, image_np)
        elapsed = time.perf_counter() - start

        try:
            prometheus_metrics.SCORING_DURATION_HIST.observe(elapsed)
            prometheus_metrics.SCORING_DURATION.set(elapsed)
            prometheus_metrics.SCORING_VALUE.set(score)
        except Exception as e:
            logger.exception(e)

        logger.info("Score: %.4f took %.2f seconds for ndarray upload, size %d bytes", score, elapsed, len(data))
    except Exception as e:
        prometheus_metrics.ERROR_COUNT.labels(type="inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    resp_filename = filename_from if filename_from else None
    response = ScoringResponsePayloadV1(
        **{"score": score,
           "filename": resp_filename,
           "message": "Scoring successful"})
    return response


# Registriere den v1 router. Weitere Router definieren mit ähnlichem Muster.
app.include_router(v1)


# Optional: Fallback-Route auf /score die Accept-Header prüft und weiterleitet. Im code gelassen für spätere Erweiterung.
# @app.post("/score")
# async def score_negotiation(request: Request, accept: str | None = Header(None)):
#     if accept and "vnd.juror.v2" in accept:
#         return await app.router.app.scope  # kurz: in real impl. an v2 handler weiterleiten
#     return Response(status_code=406, content="Specify API version in path or Accept header")

@app.get("/metrics")
async def metrics():
    """Prometheus Metriken verfügbar machen"""
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    # Note: running this module directly will start an uvicorn server.
    # Use: python -m src.juror-service.JurorServer (depending on PYTHONPATH) or run via an entry point.
    import uvicorn

    uvicorn.run("src.juror_server.JurorServer:app", host="0.0.0.0", port=8000, log_level="info")
