import base64
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional
import numpy as np
from PIL import Image
import io
from contextlib import asynccontextmanager

# Import the Juror implementation from the same package
from juror.Juror import Juror
from juror_shared.ScoringRequestPayload import ScoringRequestPayload
from juror_shared.ScoringResponsePayload import ScoringResponsePayload

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
        yield
    finally:
        _juror = None


app = FastAPI(title="Juror Inference API", lifespan=lifespan)

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


@app.post("/score", response_model=ScoringResponsePayload)
async def score_image_base64(payload: ScoringRequestPayload):
    try:
        data = base64.b64decode(payload.b64)
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        image_np = np.array(pil_img).astype(np.uint8)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {e}")

    if _juror is None:
        raise HTTPException(status_code=503, detail="Juror model not loaded yet")

    try:
        start_scoring = time.time()
        score = float(_juror.inference(image_np))
        end_scoring = time.time()
        print(f"Score: {score} took {end_scoring - start_scoring:.2f} seconds for image with length {len(data)} bytes")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # return PlainTextResponse(f"{score}")
    response = ScoringResponsePayload(**{"score": score, "filename": payload.filename, "message": "Scoring successful"})
    return response


if __name__ == "__main__":
    # Note: running this module directly will start an uvicorn server.
    # Use: python -m src.juror-service.JurorServer (depending on PYTHONPATH) or run via an entry point.
    import uvicorn

    uvicorn.run("src.juror-service.JurorServer:app", host="0.0.0.0", port=8000, log_level="info")
