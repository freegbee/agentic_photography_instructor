from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
from webapps.annotations_browser.index import AnnotationsIndex
from webapps.annotations_browser.api import router as api_router
from fastapi.responses import FileResponse

# Create FastAPI app
app = FastAPI(title="Annotations Browser")

# Mount static files under /static
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")

# root route serves the index.html
@app.get("/")
async def root_index():
    index_path = static_dir / 'index.html'
    if index_path.exists():
        return FileResponse(path=str(index_path), media_type='text/html')
    return {"status": "ok"}

# Configuration from environment
IMAGE_VOLUME_PATH = os.environ.get("IMAGE_VOLUME_PATH")
ANNOTATIONS_SEARCH_ROOT = os.environ.get("ANNOTATIONS_SEARCH_ROOT")

if IMAGE_VOLUME_PATH is None:
    # Keep app running but index won't load until env is set; API will return empty results
    image_root = None
else:
    image_root = Path(IMAGE_VOLUME_PATH)

search_root = Path(ANNOTATIONS_SEARCH_ROOT) if ANNOTATIONS_SEARCH_ROOT else None

# Ensure annotations_index is always present on app.state
try:
    _default_idx = AnnotationsIndex(image_root=Path('.') if image_root is None else image_root, search_root=search_root)
    app.state.annotations_index = _default_idx
    app.state.annotations_index_summary = {"loaded_files": [], "loaded_images": 0, "skipped_images": []}
except Exception:
    # last resort
    app.state.annotations_index = None
    app.state.annotations_index_summary = {"loaded_files": [], "loaded_images": 0, "skipped_images": []}


# Initialize index and store on app.state for access in routers
# Do NOT automatically load annotation files on startup. The UI will request
# the available annotation files and call the load endpoint for the chosen file.
_default_idx = AnnotationsIndex(image_root=Path('.') if image_root is None else image_root, search_root=search_root)
app.state.annotations_index = _default_idx
app.state.annotations_index_summary = {"loaded_files": [], "loaded_images": 0, "skipped_images": []}

# include API router
app.include_router(api_router)
