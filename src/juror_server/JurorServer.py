import base64
import logging
import time

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from typing import Optional
import numpy as np
from PIL import Image
import io
from contextlib import asynccontextmanager

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Import the Juror implementation from the same package
from juror.Juror import Juror
from juror_shared.models_v1 import ScoringRequestPayloadV1, ScoringResponsePayloadV1

logger = logging.getLogger(__name__)

# Global juror-service instance (initialized in lifespan)
_juror: Optional[Juror] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to initialize and cleanup the Juror instance."""
    from juror_server.prometheus.Metrics import JUROR_LOADED
    global _juror
    if _juror is None:
        # instantiate the juror-service (this may load a model and take time)
        print("Loading Juror model...")
        _juror = Juror()
        try:
            JUROR_LOADED.set(1)
        except Exception:
            pass
    try:
        yield
    finally:
        _juror = None
        try:
            JUROR_LOADED.set(0)
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


@app.get("/health")
async def health():
    return {"status": "ok", "juror_loaded": _juror is not None}


@v1.post("/score", response_model=ScoringResponsePayloadV1)
async def score_image_base64(payload: ScoringRequestPayloadV1):
    from juror_server.prometheus.Metrics import REQUEST_COUNT, ERROR_COUNT, SCORING_DURATION
    # Record request start for counting / metrics
    method = "POST"
    endpoint = "/v1/score"

    try:
        data = base64.b64decode(payload.b64)
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        image_np = np.array(pil_img).astype(np.uint8)
    except Exception as e:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="400").inc()
        ERROR_COUNT.labels(type="decode_error").inc()
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {e}")

    if _juror is None:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="503").inc()
        ERROR_COUNT.labels(type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Juror model not loaded yet")

    try:
        start = time.perf_counter()
        score = float(_juror.inference(image_np))
        elapsed = time.perf_counter() - start

        # Metric für Inferenzzeit und erfolgreiche Anfrage
        try:
            SCORING_DURATION.observe(elapsed)
        except Exception:
            pass

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="200").inc()
        print(f"Score: {score} took {elapsed:.2f} seconds for image with length {len(data)} bytes")
    except Exception as e:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="500").inc()
        ERROR_COUNT.labels(type="inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # return PlainTextResponse(f"{score}")
    response = ScoringResponsePayloadV1(**{"score": score, "filename": payload.filename, "message": "Scoring successful"})
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
    """Prometheus metriken verfügbar machen"""
    data = generate_latest()
    logger.debug(f"Serving /metrics data {data}")
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    # Note: running this module directly will start an uvicorn server.
    # Use: python -m src.juror-service.JurorServer (depending on PYTHONPATH) or run via an entry point.
    import uvicorn

    uvicorn.run("src.juror-service.JurorServer:app", host="0.0.0.0", port=8000, log_level="info")
