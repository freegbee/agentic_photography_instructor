import os, sys
sys.path.insert(0, r'C:\Users\holger\CASML4SE\repos\agentic_photography_instructor' if False else r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor')
from webapps.annotations_browser.index import AnnotationsIndex
from pathlib import Path

repo = Path(r'C:/Users/holge/CASML4SE/repos/agentic_photography_instructor')
idx = AnnotationsIndex(image_root=repo, search_root=repo)
print('IMAGE_ROOT', repo)
files = idx.list_annotation_files()
print('Found annotation files:', files)
res = idx.load_file('annotations_test/annotations.json')
print('Load result:', res)
print('Loaded images keys:', list(idx.images_by_id.keys()))
for k,v in idx.images_by_id.items():
    print('image id', k, 'file_name', v['file_name'], 'abs', v['abs_path'])

# try resolving sample path
rp = 'resources/sample_images/Places365_test_00000001.jpg'
print('resolve', rp, '->', idx.resolve_relative_path(rp))

