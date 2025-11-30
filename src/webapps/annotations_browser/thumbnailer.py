from pathlib import Path
from PIL import Image
import hashlib
from typing import Optional

CACHE_DIR = Path('temp_output') / 'thumbnails'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key_for(abs_path: Path, max_size: int) -> str:
    rel = abs_path.as_posix()
    key = f"{rel}:{max_size}"
    h = hashlib.sha1(key.encode('utf-8')).hexdigest()
    return h


def _atomic_write(final_path: Path, tmp_path: Path):
    # atomic replace
    tmp_path.replace(final_path)


def get_or_create_thumbnail(abs_path: Path, max_size: int = 256) -> Optional[Path]:
    """Return path to cached thumbnail PNG. Create if missing."""
    if not abs_path.exists():
        return None
    cache_key = _cache_key_for(abs_path, max_size)
    out_path = CACHE_DIR / f"{cache_key}.png"
    if out_path.exists():
        return out_path

    try:
        with Image.open(abs_path) as im:
            im = im.convert('RGB')
            # use new Resampling enum if available
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                try:
                    resample = Image.LANCZOS
                except Exception:
                    resample = Image.BICUBIC
            im.thumbnail((max_size, max_size), resample)
            tmp = CACHE_DIR / f"{cache_key}.tmp"
            im.save(tmp, format='PNG')
            _atomic_write(out_path, tmp)
        return out_path
    except Exception:
        return None
