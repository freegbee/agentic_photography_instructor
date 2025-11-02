import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter

from prometheus_client import CollectorRegistry, gc_collector, platform_collector, process_collector

from image_acquisition.acquisition_server.prometheus.Metrics import init_metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

prometheus_registry = CollectorRegistry()
# Default collectors registrieren
gc_collector.GCCollector(registry=prometheus_registry)
platform_collector.PlatformCollector(registry=prometheus_registry)
process_collector.ProcessCollector(registry=prometheus_registry)

prometheus_metrics = init_metrics(registry=prometheus_registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to initialize and cleanup resources."""
    logger.info("Starting Acquisition Server...")
    try:
        yield
    finally:
        logger.info("Shutting down Acquisition Server...")


app = FastAPI(title="Image Acquisition Server", lifespan=lifespan)

v1 = APIRouter(prefix="/v1")

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
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("acquisition_server.AcquisitionServer:app", host="0.0.0.0", port=8000, log_level="debug")