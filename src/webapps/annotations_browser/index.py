import json
from pathlib import Path
from typing import Optional, List, Dict, Union
from collections import defaultdict


class AnnotationsIndex:
    """Loads COCO-style annotations.json files and provides query capabilities.

    Only images referenced in loaded COCO files and physically present under `image_root` are indexed.
    """

    def __init__(self, image_root: Path, search_root: Optional[Path] = None):
        self.image_root = Path(image_root)
        self.search_root = Path(search_root) if search_root is not None else None
        # core structures
        self.images_by_id: Dict[int, dict] = {}
        self.annotations_by_image_id: Dict[int, List[dict]] = defaultdict(list)
        self.category_id_to_name: Dict[int, str] = {}
        self.category_name_to_id: Dict[str, int] = {}
        self.loaded_files: List[Path] = []
        self.skipped_images: List[Dict[str, str]] = []

    def _find_annotation_files(self) -> List[Path]:
        roots = []
        if self.search_root is not None:
            roots = [self.search_root]
        else:
            roots = [self.image_root]

        found = []
        for r in roots:
            for p in Path(r).rglob('annotations.json'):
                found.append(p)
        return found

    def list_annotation_files(self) -> List[str]:
        """Return list of annotation.json file paths (as strings) under search_root or image_root."""
        files = self._find_annotation_files()
        # return paths as strings (relative to image_root if possible)
        out = []
        for p in files:
            try:
                # prefer relative path to image_root for UI clarity
                try:
                    rel = p.relative_to(self.image_root)
                    out.append(str(rel).replace('\\','/'))
                except Exception:
                    out.append(str(p))
            except Exception:
                out.append(str(p))
        return out

    def load(self) -> Dict:
        """Deprecated: use load_file(file_path) or reload(file_path). Kept for backward compatibility.
        Loads nothing by default and returns empty summary.
        """
        self.images_by_id.clear()
        self.annotations_by_image_id.clear()
        self.category_id_to_name.clear()
        self.category_name_to_id.clear()
        self.loaded_files = []
        self.skipped_images = []

        return {
            "loaded_files": [],
            "loaded_images": 0,
            "skipped_images": [],
        }

    def load_file(self, file_path: Union[str, Path], progress_callback=None) -> Dict:
        """Load a single annotations.json file specified by file_path.

        file_path may be absolute or relative to search_root/image_root.
        Returns summary like load used to do.
        """
        self.images_by_id.clear()
        self.annotations_by_image_id.clear()
        self.category_id_to_name.clear()
        self.category_name_to_id.clear()
        self.loaded_files = []
        self.skipped_images = []

        # Normalize input: accept urlencoded, tilde, relative or absolute paths
        if isinstance(file_path, bytes):
            file_path = file_path.decode('utf-8')
        # strip quotes and whitespace
        fp_str = str(file_path).strip().strip('"').strip("'")
        try:
            from urllib.parse import unquote
            fp_str = unquote(fp_str)
        except Exception:
            pass
        fp = Path(fp_str)
        if not fp.is_absolute():
            # try relative to search_root if set else image_root
            base = (self.search_root if self.search_root is not None else self.image_root)
            fp = (base / fp)
        # resolve canonical path
        try:
            fp = fp.resolve()
        except Exception:
            # fallback: keep as is
            pass

        # ensure file is under image_root for safety
        try:
            if not fp.exists():
                return {"loaded_files": [], "loaded_images": 0, "skipped_images": [{"file_name": str(file_path), "reason": "file not found"}]}
            try:
                fp.relative_to(self.image_root)
            except Exception:
                # if file is outside image_root, reject
                return {"loaded_files": [], "loaded_images": 0, "skipped_images": [{"file_name": str(file_path), "reason": "outside image_root"}]}
        except Exception:
            return {"loaded_files": [], "loaded_images": 0, "skipped_images": [{"file_name": str(file_path), "reason": "invalid path"}]}

        # initialize progress counters to ensure they exist for callbacks and cancellation checks
        processed_images = 0
        total_images_est = 0

        try:
            with open(fp, 'r', encoding='utf-8') as fh:
                coco = json.load(fh)
            self.loaded_files.append(fp)

            # prepare total count for progress reporting
            total_images_est = len(coco.get('images', [])) if isinstance(coco.get('images', []), list) else 0
            processed_images = 0

            # load categories
            for c in coco.get('categories', []):
                cid = c.get('id')
                name = c.get('name')
                if cid is not None and name is not None:
                    self.category_id_to_name[cid] = name
                    self.category_name_to_id[name] = cid

            # load images
            # helper to probe cancellation via progress_callback
            def _maybe_check_cancel():
                if progress_callback:
                    # progress_callback may raise LoadCancelled
                    progress_callback(processed_images, total_images_est)

            for img in coco.get('images', []):
                img_id = img.get('id')
                try:
                    img_id = int(img_id)
                except Exception:
                    # skip invalid id
                    continue
                file_name = img.get('file_name')
                width = img.get('width', 0)
                height = img.get('height', 0)
                if img_id is None or file_name is None:
                    continue
                abs_path = None
                p = Path(file_name)
                # quick cancellation probe before filesystem ops
                try:
                    _maybe_check_cancel()
                except Exception:
                    # propagate cancellation
                    raise
                if p.is_absolute():
                    try:
                        p.relative_to(self.image_root)
                        abs_path = p.resolve()
                    except Exception:
                        self.skipped_images.append({"file_name": str(file_name), "reason": "outside image_root"})
                        continue
                else:
                    candidate = self.image_root / file_name
                    if candidate.exists():
                        try:
                            abs_path = candidate.resolve()
                        except Exception:
                            abs_path = candidate
                    else:
                        # iterate rglob to allow cancellation checks between iterations
                        abs_path = None
                        try:
                            for m in self.image_root.rglob(p.name):
                                # check cancellation between iterations
                                try:
                                    _maybe_check_cancel()
                                except Exception:
                                    raise
                                abs_path = m
                                # take first match and stop
                                break
                        except Exception:
                            # if cancellation or other exception occurred, re-raise
                            raise
                        if abs_path:
                            try:
                                abs_path = abs_path.resolve()
                            except Exception:
                                pass
                        else:
                            self.skipped_images.append({"file_name": str(file_name), "reason": "file missing"})
                            continue

                # normalize file_name to posix style for consistent matching
                file_name_norm = str(file_name).replace('\\','/')

                self.images_by_id[img_id] = {
                    "id": img_id,
                    "file_name": file_name_norm,
                    "abs_path": abs_path,
                    "width": width,
                    "height": height,
                }
                # report progress for each image processed
                processed_images += 1
                if progress_callback:
                    try:
                        progress_callback(processed_images, total_images_est)
                    except LoadCancelled:
                        raise
                    except Exception:
                        # ignore other callback errors
                        pass

            # load annotations
            ann_processed = 0
            total_anns_est = len(coco.get('annotations', [])) if isinstance(coco.get('annotations', []), list) else 0
            for ann in coco.get('annotations', []):
                image_id_raw = ann.get('image_id')
                if image_id_raw is None:
                    continue
                try:
                    image_id = int(image_id_raw)
                except Exception:
                    # skip annotation with invalid image_id
                    self.skipped_images.append({"file_name": str(image_id_raw), "reason": "annotation invalid image_id"})
                    continue

                # normalize category_id inside annotation if present
                if 'category_id' in ann:
                    try:
                        catv = ann.get('category_id')
                        if catv is not None:
                            ann['category_id'] = int(catv)
                    except Exception:
                        # leave as is if cannot convert
                        pass

                if image_id not in self.images_by_id:
                    self.skipped_images.append({"file_name": str(image_id), "reason": "annotation for unknown image"})
                    continue
                self.annotations_by_image_id[image_id].append(ann)
                ann_processed += 1
                # report annotation-level progress (allows cancellation during annotation processing)
                if progress_callback:
                    try:
                        progress_callback(processed_images, total_images_est)
                    except LoadCancelled:
                        raise
                    except Exception:
                        # ignore other callback errors
                        pass

        except LoadCancelled:
             # propagate cancellation to caller
             raise
        except Exception as e:
            self.skipped_images.append({"file_name": str(fp), "reason": f"failed to load: {e}"})

        # compute image-level score fields from annotations with category_id == 0
        for img_id, img in list(self.images_by_id.items()):
            anns = self.annotations_by_image_id.get(img_id, [])
            img_score = None
            img_initial = None

            def _get_num(d, keys):
                # prefer direct keys, allow numeric conversion and comma decimal
                for k in keys:
                    if k in d:
                        try:
                            return float(d[k])
                        except Exception:
                            try:
                                return float(str(d[k]).replace(',','.'))
                            except Exception:
                                return None
                # case-insensitive keys
                for kk, vv in d.items():
                    if isinstance(kk, str) and kk.lower() in [k.lower() for k in keys]:
                        try:
                            return float(vv)
                        except Exception:
                            try:
                                return float(str(vv).replace(',','.'))
                            except Exception:
                                return None
                return None

            for a in anns:
                try:
                    cat = a.get('category_id')
                except Exception:
                    cat = None
                if cat == 0:
                    if img_score is None:
                        img_score = _get_num(a, ['score', 'final_score', 'value'])
                    if img_initial is None:
                        img_initial = _get_num(a, ['initial_score', 'initial', 'initialValue', 'initial_value'])

                # allow cancellation check during score computation
                if progress_callback:
                    try:
                        progress_callback(processed_images, total_images_est)
                    except LoadCancelled:
                        raise
                    except Exception:
                        # ignore other callback errors
                        pass

            # attach to image meta if values found
            if img_score is not None:
                self.images_by_id[img_id]['score'] = img_score
            if img_initial is not None:
                self.images_by_id[img_id]['initial_score'] = img_initial
            if (img_score is not None) and (img_initial is not None):
                try:
                    self.images_by_id[img_id]['change'] = float(img_score) - float(img_initial)
                except Exception:
                    self.images_by_id[img_id]['change'] = None

        # Ensure that every image has the keys 'score', 'initial_score' and 'change' (may be None)
        for img_id, img in self.images_by_id.items():
            if 'score' not in img:
                img['score'] = None
            if 'initial_score' not in img:
                img['initial_score'] = None
            if 'change' not in img:
                img['change'] = None

        return {
            "loaded_files": [str(p) for p in self.loaded_files],
            "loaded_images": len(self.images_by_id),
            "skipped_images": self.skipped_images,
        }

    def get_categories(self) -> List[Dict]:
        """Return list of categories as dicts with 'id' and 'name'."""
        out = []
        for cid, name in self.category_id_to_name.items():
            out.append({"id": cid, "name": name})
        return out

    def query(self, category: Optional[Union[int, str]] = None, annotation_text: Optional[str] = None,
              sort: str = 'file_name', order: str = 'asc', page: int = 1, page_size: int = 20) -> Dict:
        """Filter, sort and paginate indexed images.

        Returns dict with total, page, page_size, items (image_id, file_name, meta)
        """
        # validate
        try:
            page = int(page)
            page_size = int(page_size)
        except Exception:
            raise ValueError("page and page_size must be integers")
        if page < 1:
            raise ValueError("page must be >= 1")
        page_size = max(1, min(100, page_size))

        items_list = []
        for img_id, img in self.images_by_id.items():
            include = True
            anns = self.annotations_by_image_id.get(img_id, [])

            # category filter
            if category is not None:
                cat_id = None
                if isinstance(category, int) or (isinstance(category, str) and category.isdigit()):
                    try:
                        cat_id = int(category)
                    except Exception:
                        cat_id = None
                if cat_id is None:
                    cat_id = self.category_name_to_id.get(str(category))
                if cat_id is None:
                    include = False
                else:
                    found = False
                    for a in anns:
                        if a.get('category_id') == cat_id:
                            found = True
                            break
                    if not found:
                        include = False

            # annotation_text search
            if annotation_text and include:
                at = str(annotation_text).lower()
                found = False
                for a in anns:
                    for v in a.values():
                        if v is None:
                            continue
                        if isinstance(v, (int, float)):
                            s = str(v)
                        else:
                            s = str(v)
                        if at in s.lower():
                            found = True
                            break
                    if found:
                        break
                if not found:
                    include = False

            if include:
                items_list.append(img)

        # sort
        reverse = (order == 'desc')
        if sort == 'file_name':
            items_list.sort(key=lambda x: x.get('file_name', ''), reverse=reverse)
        elif sort == 'image_id':
            items_list.sort(key=lambda x: x.get('id', 0), reverse=reverse)
        elif sort == 'width':
            items_list.sort(key=lambda x: x.get('width', 0), reverse=reverse)
        elif sort == 'height':
            items_list.sort(key=lambda x: x.get('height', 0), reverse=reverse)
        elif sort == 'score':
            def _score_key(x):
                img_id = x.get('id')
                anns = self.annotations_by_image_id.get(img_id, [])
                final = None
                for a in anns:
                    if a.get('category_id') in (0, None):
                        if 'score' in a:
                            final = a.get('score')
                return final if final is not None else 0
            items_list.sort(key=_score_key, reverse=reverse)
        elif sort == 'change':
            # Sort by change (score - initial_score from category==0). Missing values should always
            # be placed at the end regardless of asc/desc. To achieve that we return a tuple
            # (missing_flag, adjusted_value) where missing_flag==1 for missing values so they
            # sort after present values; adjusted_value is negated for descending order.
            def _change_key(x):
                # prefer precomputed image-level 'change'
                val = None
                try:
                    img = self.images_by_id.get(x.get('id'))
                    if img and ('change' in img) and (img.get('change') is not None):
                        val = float(img.get('change'))
                except Exception:
                    val = None
                # fallback: compute from annotations category==0
                if val is None:
                    img_id = x.get('id')
                    anns = self.annotations_by_image_id.get(img_id, [])
                    score = None
                    initial = None
                    for a in anns:
                        try:
                            if a.get('category_id') in (0, None):
                                if score is None and 'score' in a:
                                    try:
                                        score = float(a.get('score'))
                                    except Exception:
                                        try:
                                            score = float(str(a.get('score')).replace(',','.'))
                                        except Exception:
                                            score = None
                                if initial is None:
                                    for k in ('initial_score','initial','initialValue','initial_value'):
                                        if k in a:
                                            try:
                                                initial = float(a.get(k))
                                            except Exception:
                                                try:
                                                    initial = float(str(a.get(k)).replace(',','.'))
                                                except Exception:
                                                    initial = None
                                            if initial is not None:
                                                break
                        except Exception:
                            continue
                    if score is not None and initial is not None:
                        try:
                            val = float(score) - float(initial)
                        except Exception:
                            val = None

                missing = 1 if (val is None or (isinstance(val, float) and (val != val))) else 0
                # adjusted value: negate when descending so we can sort with reverse=False and
                # still keep missing values at the end
                adjusted = val if not reverse else (-val if val is not None else 0.0)
                return (missing, adjusted)

            # Use reverse=False intentionally; we've encoded order into adjusted value so missing
            # always sorts to the end (missing==1).
            items_list.sort(key=_change_key)
        else:
            items_list.sort(key=lambda x: x.get('file_name', ''), reverse=reverse)

        total = len(items_list)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = items_list[start:end]

        result_items = []
        for img in page_items:
            img_id = img.get('id')
            # per-image level score data (may be absent)
            score = img.get('score') if img.get('score') is not None else None
            initial_score = img.get('initial_score') if img.get('initial_score') is not None else None
            change = img.get('change') if img.get('change') is not None else None

            # build per-category metadata from annotations for this image
            cat_map = {}
            anns = self.annotations_by_image_id.get(img_id, [])
            def _to_num(v):
                try:
                    return float(v)
                except Exception:
                    try:
                        return float(str(v).replace(',','.'))
                    except Exception:
                        return None

            for a in anns:
                cid = a.get('category_id')
                if cid is None:
                    continue
                if cid not in cat_map:
                    cat_map[cid] = {'id': cid, 'name': self.category_id_to_name.get(cid), 'score': None, 'initial_score': None}
                # try find score/initial in annotation
                if cat_map[cid]['score'] is None:
                    s = None
                    for k in ('score','final_score','value'):
                        if k in a:
                            s = _to_num(a.get(k))
                            if s is not None:
                                break
                    cat_map[cid]['score'] = s
                if cat_map[cid]['initial_score'] is None:
                    isv = None
                    for k in ('initial_score','initial','initialValue','initial_value'):
                        if k in a:
                            isv = _to_num(a.get(k))
                            if isv is not None:
                                break
                    cat_map[cid]['initial_score'] = isv

            categories_meta = []
            for cid, info in cat_map.items():
                sc = info.get('score')
                ic = info.get('initial_score')
                ch = None
                if sc is not None and ic is not None:
                    try:
                        ch = float(sc) - float(ic)
                    except Exception:
                        ch = None
                categories_meta.append({'id': cid, 'name': info.get('name'), 'score': sc, 'initial_score': ic, 'change': ch})

            result_items.append({
                'image_id': img_id,
                'file_name': img.get('file_name'),
                'meta': {
                    'width': img.get('width', 0),
                    'height': img.get('height', 0),
                    'categories': [ { 'id': a.get('category_id'), 'name': self.category_id_to_name.get(a.get('category_id')) } for a in anns if a.get('category_id') is not None ],
                    'scores': [ a.get('score') for a in anns if 'score' in a ],
                    'score': score,
                    'initial_score': initial_score,
                    'change': change,
                    'categories_meta': categories_meta
                }
            })

        return {
            'total': total,
            'page': page,
            'page_size': page_size,
            'items': result_items
        }

    def resolve_relative_path(self, rel_path: str) -> Optional[Path]:
        """Resolve a relative path (as appears in annotations) to an absolute Path under image_root.
        Returns Path if found and exists, otherwise None.
        """
        try:
            p = Path(rel_path)
            if p.is_absolute():
                # ensure under image_root
                try:
                    p.relative_to(self.image_root)
                    return p.resolve()
                except Exception:
                    return None
            # try direct candidate under image_root
            candidate = self.image_root / rel_path
            if candidate.exists():
                try:
                    return candidate.resolve()
                except Exception:
                    return candidate
            # fall back to searching by name
            name = p.name
            for m in self.image_root.rglob(name):
                try:
                    return m.resolve()
                except Exception:
                    return m
        except Exception:
            return None


class LoadCancelled(Exception):
    """Raised when a load operation was cancelled by the caller."""
    pass
