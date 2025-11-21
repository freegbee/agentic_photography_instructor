import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter

from prometheus_client import CollectorRegistry, CONTENT_TYPE_LATEST, gc_collector, platform_collector, process_collector

from image_acquisition.acquisition_server.JobManager import JobManager
from image_acquisition.acquisition_server.ImageAcquisitionJob import ImageAcquisitionJob
from utils.LoggingUtils import configure_logging
from image_acquisition.acquisition_server.prometheus import prometheus_metrics
from image_acquisition.acquisition_shared.models_v1 import StartAsyncImageAcquisitionRequestV1, \
    AsyncImageAcquisitionJobResponseV1

logger = logging.getLogger(__name__)

prometheus_registry = CollectorRegistry()
# Default collectors registrieren
gc_collector.GCCollector(registry=prometheus_registry)
platform_collector.PlatformCollector(registry=prometheus_registry)
process_collector.ProcessCollector(registry=prometheus_registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to initialize and cleanup resources."""
    listener = configure_logging()
    logger.info("Starting Acquisition Server...")

    # Defensive: ensure app.state exists (manche Checker / Umgebungen melden es sonst nicht)
    if not hasattr(app, "state") or getattr(app, "state") is None:
        app.state = SimpleNamespace()  # sicherer, leichter Behälter für Attribute

    try:
        # Listener verfügbar machen, falls andere Teile der App darauf zugreifen wollen
        app.state.log_listener = listener
        yield
    finally:
        logger.info("Shutting down Acquisition Server...")
        # Listener sicher stoppen
        try:
            if getattr(app.state, "log_listener", None):
                app.state.log_listener.stop()
        except Exception:
            logger.exception("Error while stopping logging listener")


app = FastAPI(title="Image Acquisition Server", lifespan=lifespan)

v1 = APIRouter(prefix="/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton-Instanz einmalig holen
job_manager: JobManager = JobManager.get_instance()

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware: läuft vor und nach jedem Request; inkrementiert Request-Counter nach Antwort."""
    start = time.perf_counter()
    response = await call_next(request)  # führt die Route aus und liefert die Response zurück
    elapsed = time.perf_counter() - start

    # Metriken für bestimmte Endpunkte überspringen
    try:
        if request.url.path in ("/metrics", "/health"):
            return response

        logger.debug("Request to %s tool %s seconds", request.url.path, elapsed)
        prometheus_metrics.metrics().HTTP_REQUEST_DURATION_HIST.labels(
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

@app.get("/metrics")
async def metrics():
    """Prometheus Metriken verfügbar machen"""
    return Response(content=prometheus_metrics.generate_latest(), media_type=CONTENT_TYPE_LATEST)


@v1.post("/acquisition", response_model=AsyncImageAcquisitionJobResponseV1)
async def start_acquisition(request: StartAsyncImageAcquisitionRequestV1):
    """Startet einen asynchronen Task, um Bilder zu akquirieren."""
    logger.info("Request für image acquisition task...")
    job_uuid = str(uuid.uuid4())
    try:
        new_job = ImageAcquisitionJob(job_uuid, request.dataset_id)
    except ValueError as e:
        print(f"Config {request.dataset_id} not found...")
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not available.")

    try:
        job_manager.add_job(new_job)
    except KeyError as ke:
        raise HTTPException(status_code=409, detail=f"Job with UUID {new_job.uuid} already exists.")
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))

    # Starte den asynchronen Task
    asyncio.create_task(_run_image_acquisition_job(new_job))
    response = AsyncImageAcquisitionJobResponseV1(**{"job_uuid": new_job.uuid, "status": new_job.status})
    logger.info("Started async job with UUID %s for image acquisition of dataset %s.", new_job.uuid, new_job.dataset_id)
    return response

@v1.get("/acquisition/jobs/{job_uuid}", response_model=AsyncImageAcquisitionJobResponseV1)
async def get_acquisition_job(job_uuid: str):
    """Gibt den Status eines asynchronen Image Acquisition Jobs zurück."""

    try:
        job = job_manager.get_job(job_uuid)
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=f"Job with UUID {job_uuid} not found.")
    return AsyncImageAcquisitionJobResponseV1(**{"job_uuid": job.uuid, "status": job.status, "resulting_hash": job.resulting_hash})

app.include_router(v1)

async def _run_image_acquisition_job(job: ImageAcquisitionJob):
    logger.info("Running image acquisition job %s...", job.uuid)
    try:
        # job.start() ist synchron -> in Thread auslagern, damit Event-Loop nicht blockiert
        await asyncio.to_thread(job.start)
    except Exception as e:
        logger.exception(f"Error running job %s: %s", {job.uuid}, e)
    finally:
        logger.info(f"Finished image acquisition job %s with status %s.", job.uuid, job.status)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("acquisition_server.AcquisitionServer:app", host="0.0.0.0", port=8000, log_level="debug")