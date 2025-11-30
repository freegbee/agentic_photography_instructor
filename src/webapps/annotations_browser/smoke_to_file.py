import os, sys, json
sys.path.insert(0, r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor')
os.environ['IMAGE_VOLUME_PATH'] = r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor'
from webapps.annotations_browser.app import app
from fastapi.testclient import TestClient
outp = []
with TestClient(app) as client:
    r = client.get('/api/list_annotation_files')
    outp.append(('list', r.status_code, r.json()))
    r = client.post('/api/load_annotations', json={'file_path':'annotations_test/annotations.json'})
    outp.append(('load', r.status_code, r.json()))
    r = client.get('/api/categories')
    outp.append(('categories', r.status_code, r.json()))
    r = client.get('/api/images')
    outp.append(('images', r.status_code, r.json()))
    imgs = outp[-1][2].get('items', []) if isinstance(outp[-1][2], dict) else []
    if imgs:
        first = imgs[0]['image_id']
        r = client.get(f'/api/thumbnail?id={first}&max_size=64')
        outp.append(('thumbnail', r.status_code, r.headers.get('content-type')))
    else:
        outp.append(('thumbnail', None, 'no images'))

# write to file
p = Path = __import__('pathlib').Path
fpath = p('temp_output')
fpath.mkdir(parents=True, exist_ok=True)
(outp_file := fpath / 'smoke_result.json').write_text(json.dumps(outp, default=str, indent=2))
print('wrote', str(outp_file))

