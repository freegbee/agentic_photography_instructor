from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from pathlib import Path
from .index import AnnotationsIndex
from .thumbnailer import get_or_create_thumbnail
import urllib.parse

router = APIRouter(prefix="/api")


@router.get('/images')
async def get_images(request: Request, category: Optional[str] = None, annotation: Optional[str] = None,
                     sort: str = 'file_name', order: str = 'asc', page: int = 1, page_size: int = 20):
    idx: AnnotationsIndex = request.app.state.annotations_index
    try:
        res = idx.query(category=category, annotation_text=annotation, sort=sort, order=order, page=page, page_size=page_size)
        # encode URLs properly
        for it in res.get('items', []):
            img_id = it.get('image_id')
            # prefer id-based URLs (more robust); also include path-based fallback
            fn = it.get('file_name')
            it['thumbnail_url'] = f"/api/thumbnail?id={img_id}&max_size=256"
            it['image_url'] = f"/api/image?id={img_id}"
        return JSONResponse(content=res)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/thumbnail')
async def thumbnail(request: Request, path: Optional[str] = None, id: Optional[int] = None, max_size: int = 256):
    idx: AnnotationsIndex = request.app.state.annotations_index
    abs_path = None
    if id is not None:
        try:
            img = idx.images_by_id.get(int(id))
            if img:
                abs_path = img.get('abs_path')
        except Exception:
            abs_path = None
    if abs_path is None and path is not None:
        rel = urllib.parse.unquote(path)
        abs_path = idx.resolve_relative_path(rel)

    if abs_path is None:
        raise HTTPException(status_code=403, detail="Path or id not referenced in annotations")
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if max_size <= 0 or max_size > 1024:
        raise HTTPException(status_code=400, detail="max_size must be between 1 and 1024")

    thumb = get_or_create_thumbnail(abs_path, max_size=max_size)
    if thumb is None:
        raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
    return FileResponse(path=thumb, media_type='image/png')


@router.get('/image')
async def image(request: Request, path: Optional[str] = None, id: Optional[int] = None):
    idx: AnnotationsIndex = request.app.state.annotations_index
    abs_path = None
    if id is not None:
        try:
            img = idx.images_by_id.get(int(id))
            if img:
                abs_path = img.get('abs_path')
        except Exception:
            abs_path = None
    if abs_path is None and path is not None:
        rel = urllib.parse.unquote(path)
        abs_path = idx.resolve_relative_path(rel)
    if abs_path is None:
        raise HTTPException(status_code=403, detail="Path or id not referenced in annotations")
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # guess content type by suffix
    suffix = abs_path.suffix.lower()
    if suffix in ['.jpg', '.jpeg']:
        media = 'image/jpeg'
    elif suffix in ['.png']:
        media = 'image/png'
    elif suffix in ['.gif']:
        media = 'image/gif'
    else:
        media = 'application/octet-stream'

    return FileResponse(path=abs_path, media_type=media)


@router.get('/categories')
async def categories(request: Request):
    idx: AnnotationsIndex = request.app.state.annotations_index
    return JSONResponse(content={"categories": idx.get_categories()})


@router.get('/list_annotation_files')
async def list_annotation_files(request: Request):
    idx: AnnotationsIndex = request.app.state.annotations_index
    files = idx.list_annotation_files()
    return JSONResponse(content={"files": files})


@router.post('/load_annotations')
async def load_annotations(request: Request):
    """Load a specific annotations.json file supplied in the request body as {"file_path": "..."}.
    file_path may be absolute or relative to search_root/image_root.
    """
    idx: AnnotationsIndex = request.app.state.annotations_index
    body = await request.json() if request.headers.get('content-type','').startswith('application/json') else {}
    file_path = body.get('file_path') if isinstance(body, dict) else None
    if not file_path:
        raise HTTPException(status_code=400, detail='file_path must be provided')
    res = idx.load_file(file_path)
    # store summary
    request.app.state.annotations_index_summary = res
    return JSONResponse(content=res)


@router.post('/reload')
async def reload(request: Request):
    idx: AnnotationsIndex = request.app.state.annotations_index
    body = await request.json() if request.headers.get('content-type','').startswith('application/json') else {}
    search_root = body.get('search_root') if isinstance(body, dict) else None
    if search_root:
        sr = Path(search_root)
        idx.search_root = sr
    res = {
        "loaded_files": idx.list_annotation_files(),
        "loaded_images": len(idx.images_by_id),
        "skipped_images": idx.skipped_images,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat() + 'Z'
    }
    request.app.state.annotations_index_summary = res
    return JSONResponse(content=res)
