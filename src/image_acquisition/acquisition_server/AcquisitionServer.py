import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter

from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, gc_collector, platform_collector, process_collector

from image_acquisition.acquisition_server.imageacquisitionjob import ImageAcquisitionJob
from image_acquisition.acquisition_server.prometheus.Metrics import init_metrics
from image_acquisition.acquisition_shared.models_v1 import StartAsyncImageAcquisitionRequestV1, \
    AsyncImageAcquisitionJobResponseV1, AsyncJobStatusV1

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

async_jobs: Dict[str, ImageAcquisitionJob] = {}

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

@app.get("/metrics")
async def metrics():
    """Prometheus Metriken verfügbar machen"""
    return Response(content=generate_latest(prometheus_registry), media_type=CONTENT_TYPE_LATEST)


@v1.post("/acquisition", response_model=AsyncImageAcquisitionJobResponseV1)
async def start_acquisition(request: StartAsyncImageAcquisitionRequestV1):
    """Startet einen asynchronen Task, um Bilder zu akquirieren."""
    logger.debug("Request für image acquisition task...")
    job_uuid = str(uuid.uuid4())
    new_job = ImageAcquisitionJob(job_uuid, request.dataset_id)
    async_jobs[job_uuid] = new_job
    logger.info(f"Started image acquisition job {new_job.uuid} for dataset {new_job.dataset_id}")

    # Starte den asynchronen Task
    asyncio.create_task(_run_image_acquisition_job(new_job))
    response = AsyncImageAcquisitionJobResponseV1(**{"job_uuid": new_job.uuid, "status": new_job.status})
    logger.info(f"Started async job with UUID {new_job.uuid} for image acquisition of dataset {new_job.dataset_id}.")
    return response

@v1.get("/acquisition/jobs/{job_uuid}", response_model=AsyncImageAcquisitionJobResponseV1)
async def get_acquisition_job(job_uuid: str):
    """Gibt den Status eines asynchronen Image Acquisition Jobs zurück."""
    job = async_jobs.get(job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with UUID {job_uuid} not found.")
    response = AsyncImageAcquisitionJobResponseV1(**{"job_uuid": job.uuid, "status": job.status})
    return response

app.include_router(v1)

async def _run_image_acquisition_job(job: ImageAcquisitionJob):
    """Simulierter asynchroner Task zur Bildakquisition."""
    logger.info(f"Running image acquisition job {job.uuid}...")
    job.status = AsyncJobStatusV1.RUNNING
    # FIXME: Dies ist nur ein Platzhalter für die eigentliche Bildakquisitionslogik
    await asyncio.sleep(10)  # Simuliere lange laufende Aufgabe
    job.status = AsyncJobStatusV1.COMPLETED
    logger.info(f"Completed image acquisition job {job.uuid}.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("acquisition_server.AcquisitionServer:app", host="0.0.0.0", port=8000, log_level="debug")