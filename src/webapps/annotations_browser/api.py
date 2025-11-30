from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from pathlib import Path
from .index import AnnotationsIndex, LoadCancelled
from .thumbnailer import get_or_create_thumbnail
import urllib.parse
import threading
import time
import multiprocessing
import tempfile
import json as _json
import os as _os

router = APIRouter(prefix="/api")


# helper to initialize load_status on app.state
def _init_load_status(app):
    if not hasattr(app.state, 'load_status') or app.state.load_status is None:
        app.state.load_status = {
            'state': 'idle',  # idle, loading, done, error
            'file': None,
            'processed': 0,
            'total': 0,
            'loaded_files': [],
            'loaded_images': 0,
            'skipped_images': [],
            'error': None,
            'timestamp': None
        }


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

    _init_load_status(request.app)
    # prevent concurrent loads
    if request.app.state.load_status.get('state') == 'loading':
        raise HTTPException(status_code=409, detail='Another load is in progress')

    # set initial status
    request.app.state.load_status.update({
        'state': 'loading',
        'file': str(file_path),
        'processed': 0,
        'total': 0,
        'loaded_files': [],
        'loaded_images': 0,
        'skipped_images': [],
        'error': None,
        'timestamp': __import__('datetime').datetime.utcnow().isoformat() + 'Z'
    })

    # initialize/clear cancel event for this load
    request.app.state.load_cancel_event = threading.Event()

    # We'll run the heavy parsing in a separate process so we can terminate it immediately on cancel.
    q = multiprocessing.Queue()

    # create a temp file path for child to write serialized index data
    tf = tempfile.NamedTemporaryFile(prefix='annotations_index_', suffix='.json', delete=False)
    tf_path = tf.name
    tf.close()

    # child target
    def _child_load(fp, image_root, search_root, out_path, q):
        # runs in separate process
        try:
            child_idx = AnnotationsIndex(image_root=Path(image_root), search_root=Path(search_root) if search_root else None)
            def child_progress(p, t):
                try:
                    q.put({'type': 'progress', 'processed': p, 'total': t})
                except Exception:
                    pass
            res = child_idx.load_file(fp, progress_callback=child_progress)
            # prepare serializable structures
            images_ser = {}
            for iid, im in child_idx.images_by_id.items():
                im_copy = dict(im)
                # convert abs_path to string
                if im_copy.get('abs_path') is not None:
                    try:
                        im_copy['abs_path'] = str(im_copy['abs_path'])
                    except Exception:
                        im_copy['abs_path'] = None
                images_ser[iid] = im_copy
            annotations_ser = {}
            for iid, al in child_idx.annotations_by_image_id.items():
                annotations_ser[iid] = al
            categories_ser = child_idx.category_id_to_name
            out = {'images_by_id': images_ser, 'annotations_by_image_id': annotations_ser, 'category_id_to_name': categories_ser, 'summary': res}
            try:
                with open(out_path, 'w', encoding='utf-8') as ofh:
                    _json.dump(out, ofh)
            except Exception as e:
                q.put({'type': 'error', 'error': f'child write failed: {e}'})
                return
            q.put({'type': 'done', 'out_path': out_path, 'summary': res})
        except Exception as e:
            try:
                q.put({'type': 'error', 'error': str(e)})
            except Exception:
                pass

    # start child process
    proc = multiprocessing.Process(target=_child_load, args=(file_path, str(idx.image_root), str(idx.search_root) if idx.search_root else '', tf_path, q), daemon=True)
    proc.start()
    request.app.state.load_process = proc
    request.app.state.load_queue = q

    # monitor loop in a thread so we don't block the request handler
    def monitor():
        try:
            while True:
                try:
                    msg = q.get(timeout=0.5)
                except Exception:
                    msg = None
                # check for cancel request
                if hasattr(request.app.state, 'load_cancel_event') and request.app.state.load_cancel_event is not None and request.app.state.load_cancel_event.is_set():
                    # terminate child immediately, then kill if still alive after short wait
                    try:
                        if proc.is_alive():
                            proc.terminate()
                            # give it a short grace period
                            proc.join(timeout=0.5)
                            if proc.is_alive():
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    request.app.state.load_status['state'] = 'cancelled'
                    request.app.state.load_status['timestamp'] = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
                    break
                if msg is None:
                    # check if process ended
                    if not proc.is_alive():
                        break
                    continue
                if msg.get('type') == 'progress':
                    request.app.state.load_status['processed'] = msg.get('processed', request.app.state.load_status.get('processed', 0))
                    request.app.state.load_status['total'] = msg.get('total', request.app.state.load_status.get('total', 0))
                    request.app.state.load_status['timestamp'] = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
                elif msg.get('type') == 'done':
                    # child finished successfully; load the serialized data
                    out_path = msg.get('out_path')
                    try:
                        with open(out_path, 'r', encoding='utf-8') as fh:
                            data = _json.load(fh)
                        # populate parent index structures
                        images_ser = data.get('images_by_id', {})
                        annotations_ser = data.get('annotations_by_image_id', {})
                        cats = data.get('category_id_to_name', {})
                        # clear existing and set
                        idx.images_by_id.clear()
                        idx.annotations_by_image_id.clear()
                        idx.category_id_to_name.clear()
                        idx.category_name_to_id.clear()
                        for k, v in images_ser.items():
                            try:
                                kk = int(k)
                            except Exception:
                                kk = int(k)
                            if v.get('abs_path') is not None:
                                try:
                                    v['abs_path'] = Path(v['abs_path'])
                                except Exception:
                                    v['abs_path'] = None
                            idx.images_by_id[kk] = v
                        for k, al in annotations_ser.items():
                            try:
                                kk = int(k)
                            except Exception:
                                kk = int(k)
                            idx.annotations_by_image_id[kk] = al
                        for cid, name in cats.items():
                            try:
                                cii = int(cid)
                            except Exception:
                                cii = int(cid)
                            idx.category_id_to_name[cii] = name
                            idx.category_name_to_id[name] = cii
                        # update summary
                        res = msg.get('summary', {})
                        request.app.state.annotations_index_summary = res
                        request.app.state.load_status['loaded_files'] = res.get('loaded_files', [])
                        request.app.state.load_status['loaded_images'] = res.get('loaded_images', 0)
                        request.app.state.load_status['skipped_images'] = res.get('skipped_images', [])
                        request.app.state.load_status['state'] = 'done'
                        request.app.state.load_status['timestamp'] = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
                    except Exception as e:
                        request.app.state.load_status['state'] = 'error'
                        request.app.state.load_status['error'] = f'failed to load child output: {e}'
                        request.app.state.load_status['timestamp'] = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
                    finally:
                        # cleanup temp file
                        try:
                            _os.remove(msg.get('out_path'))
                        except Exception:
                            pass
                    break
                elif msg.get('type') == 'error':
                    request.app.state.load_status['state'] = 'error'
                    request.app.state.load_status['error'] = msg.get('error')
                    request.app.state.load_status['timestamp'] = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
                    break
        finally:
            try:
                # ensure process is cleaned up
                if proc.is_alive():
                    proc.join(timeout=0.1)
            except Exception:
                pass
            request.app.state.load_process = None
            try:
                request.app.state.load_queue = None
            except Exception:
                pass

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    return JSONResponse(content={'status': 'started', 'file': file_path})


@router.post('/load_cancel')
async def load_cancel(request: Request):
    """Request cancellation of an ongoing load. Returns 200 when cancelled or 400 if no load in progress."""
    _init_load_status(request.app)
    st = request.app.state.load_status
    if st.get('state') != 'loading':
        raise HTTPException(status_code=400, detail='No load in progress')
    # create or set cancel event
    if not hasattr(request.app.state, 'load_cancel_event') or request.app.state.load_cancel_event is None:
        request.app.state.load_cancel_event = threading.Event()
    request.app.state.load_cancel_event.set()
    # update status
    request.app.state.load_status['state'] = 'cancelling'
    request.app.state.load_status['timestamp'] = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
    return JSONResponse(content={'status': 'cancelling'})


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


@router.get('/load_status')
async def load_status(request: Request):
    _init_load_status(request.app)
    return JSONResponse(content=request.app.state.load_status)
