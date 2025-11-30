from pathlib import Path
import os, sys
sys.path.insert(0, r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor')
from src.webapps.annotations_browser.index import AnnotationsIndex
from src.webapps.annotations_browser.thumbnailer import get_or_create_thumbnail

repo = Path(r'C:/Users/holge/CASML4SE/repos/agentic_photography_instructor')
image_root = repo
search_root = repo / 'annotations_test'

print('image_root', image_root)
print('search_root', search_root)

idx = AnnotationsIndex(image_root=image_root, search_root=search_root)
res = idx.load()
print('load result:', res)
print('images_by_id keys:', list(idx.images_by_id.keys()))

if idx.images_by_id:
    first = next(iter(idx.images_by_id.values()))
    print('first image file_name', first['file_name'], 'abs_path', first['abs_path'])
    thumb = get_or_create_thumbnail(first['abs_path'], max_size=128)
    print('thumbnail path=', thumb)
else:
    print('no images indexed')

