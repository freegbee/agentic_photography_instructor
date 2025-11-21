import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Response, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, process_collector, gc_collector, \
    platform_collector

# Import the Juror implementation from the same package
from juror.Juror import Juror
from juror_server.prometheus.Metrics import init_metrics
from juror_shared.models_v1 import ScoringRequestPayloadV1, ScoringResponsePayloadV1, ScoringNdarrayRequestPayloadV1

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to initialize and cleanup the Juror instance."""
    global _juror
    if _juror is None:
        # instantiate the juror-service (this may load a model and take time)
        print("Loading Juror model...")
        _juror = Juror()
        try:
            prometheus_metrics.JUROR_LOADED.set(1)
        except Exception:
            pass
    try:
        yield
    finally:
        _juror = None
        try:
            prometheus_metrics.JUROR_LOADED.set(0)
        except Exception:
            pass


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
    start = time.perf_counter()
    response = await call_next(request)  # führt die Route aus und liefert die Response zurück
    elapsed = time.perf_counter() - start

    print("Request to", request.url.path, "took", elapsed, "seconds")

    # Metriken für bestimmte Endpunkte überspringen
    try:
        if request.url.path in ("/metrics", "/health"):
            return response

        prometheus_metrics.HTTP_REQUEST_DURATION.labels(
            status=str(response.status_code),
            endpoint=request.url.path,
            method=request.method
        ).observe(elapsed)
    except Exception as e:
        # Fehler beim Metrik-Update dürfen Request nicht brechen
        pass

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
            prometheus_metrics.SCORING_DURATION.observe(elapsed)
            prometheus_metrics.SCORING_SPEED.set(elapsed)
            prometheus_metrics.SCORING_VALUE.set(score)
        except Exception:
            pass

        print(f"Score: {score} took {elapsed:.2f} seconds for image with length {len(data)} bytes")
    except Exception as e:
        prometheus_metrics.ERROR_COUNT.labels(type="inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # return PlainTextResponse(f"{score}")
    response = ScoringResponsePayloadV1(
        **{"score": score, "filename": payload.filename, "message": "Scoring successful"})
    return response


@v1.post("/score/ndarray", response_model=ScoringResponsePayloadV1)
async def score_ndarray_file(
    request: Request,
    metadata: Optional[str] = Form(None),
    array_file: Optional[UploadFile] = File(None),
):
    """Akzeptiert ein npy/npz File (Multipart Upload) und wertet es aus.

    Der Body sollte ein Multipart-Form sein mit Feld `array_file` (binär)
    und optional `metadata` als JSON-String (FastAPI behandelt das).
    """
    # Read raw bytes. Support two modes:
    # 1) Multipart form with `array_file` UploadFile
    # 2) Raw application/octet-stream body (request.body())
    try:
        content_type = request.headers.get("content-type", "")
        if array_file is not None:
            data = await array_file.read()
            filename_from = array_file.filename
        else:
            # If not multipart, accept raw octet-stream body
            # Note: this will also read npy/npz bytes directly
            data = await request.body()
            filename_from = None
    except Exception as e:
        prometheus_metrics.ERROR_COUNT.labels(type="read_error").inc()
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded data: {e}")

    print(f"success reading fake data of length {len(data)} bytes")

    # Load numpy array from bytes (support npy and npz)
    try:
        bio = io.BytesIO(data)
        # Try as np.load (handles .npy and .npz)
        arr = np.load(bio, allow_pickle=False)
        # If npz archive, try to extract the first array
        if isinstance(arr, np.lib.npyio.NpzFile):
            # take the first array in the archive
            keys = list(arr.files)
            if not keys:
                raise ValueError("npz archive contains no arrays")
            arr = arr[keys[0]]
        image_np = arr.astype(np.uint8)
        print(f"image_np shape: {image_np.shape}, dtype: {image_np.dtype}")
    except Exception as e:
        prometheus_metrics.ERROR_COUNT.labels(type="decode_error").inc()
        raise HTTPException(status_code=400, detail=f"Failed to decode numpy array: {e}")

    # Optional: parse and validate metadata. There are three ways metadata can arrive:
    # 1) Multipart form field `metadata` (JSON string)
    # 2) Header 'X-Scoring-Metadata' containing JSON when request is application/octet-stream
    # 3) Not provided
    metadata_obj = None
    if metadata:
        try:
            metadata_obj = ScoringNdarrayRequestPayloadV1.model_validate_json(metadata)
        except Exception as e:
            prometheus_metrics.ERROR_COUNT.labels(type="metadata_parse_error").inc()
            raise HTTPException(status_code=400, detail=f"Failed to parse metadata JSON from form field: {e}")

    # If no metadata form was provided and content type is octet-stream, try header
    if metadata is None and content_type and content_type.split(";")[0].strip() == "application/octet-stream":
        print(f"Trying to read metadata from headers")
        header_json = request.headers.get("x-scoring-metadata") or request.headers.get("x-metadata")
        if header_json:
            try:
                metadata_obj = ScoringNdarrayRequestPayloadV1.model_validate_json(header_json)
            except Exception as e:
                prometheus_metrics.ERROR_COUNT.labels(type="metadata_parse_error").inc()
                raise HTTPException(status_code=400, detail=f"Failed to parse metadata JSON from header: {e}")

    # If metadata parsed, validate shape/dtype
    if metadata_obj:
        if metadata_obj.shape and tuple(metadata_obj.shape) != image_np.shape:
            prometheus_metrics.ERROR_COUNT.labels(type="shape_mismatch").inc()
            raise HTTPException(status_code=400, detail=f"Shape mismatch: expected {metadata_obj.shape}, got {image_np.shape}")
        if metadata_obj.dtype and metadata_obj.dtype != str(image_np.dtype):
            prometheus_metrics.ERROR_COUNT.labels(type="dtype_mismatch").inc()
            raise HTTPException(status_code=400, detail=f"Dtype mismatch: expected {metadata_obj.dtype}, got {image_np.dtype}")

    if _juror is None:
        prometheus_metrics.ERROR_COUNT.labels(type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Juror model not loaded yet")

    try:
        print(f"start to score")
        start = time.perf_counter()
        score = float(_juror.inference(image_np))
        elapsed = time.perf_counter() - start

        try:
            prometheus_metrics.SCORING_DURATION.observe(elapsed)
            prometheus_metrics.SCORING_SPEED.set(elapsed)
            prometheus_metrics.SCORING_VALUE.set(score)
        except Exception:
            pass

        print(f"Score: {score} took {elapsed:.2f} seconds for ndarray upload, size {len(data)} bytes")
    except Exception as e:
        prometheus_metrics.ERROR_COUNT.labels(type="inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # determine filename for response: prefer metadata.filename > uploaded filename > None
    resp_filename = None
    if metadata_obj and metadata_obj.filename:
        resp_filename = metadata_obj.filename
    elif filename_from:
        resp_filename = filename_from

    response = ScoringResponsePayloadV1(**{"score": score, "filename": resp_filename, "message": "Scoring successful"})
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
    return Response(content=generate_latest(registry), media_type = CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    # Note: running this module directly will start an uvicorn server.
    # Use: python -m src.juror-service.JurorServer (depending on PYTHONPATH) or run via an entry point.
    import uvicorn

    uvicorn.run("src.juror-service.JurorServer:app", host="0.0.0.0", port=8000, log_level="info")
