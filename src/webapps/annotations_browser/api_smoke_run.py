import os, sys
sys.path.insert(0, r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor')
os.environ['IMAGE_VOLUME_PATH'] = r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor'

from webapps.annotations_browser.app import app
from fastapi.testclient import TestClient

print('Starting API smoke run')
with TestClient(app) as client:
    resp = client.get('/api/list_annotation_files')
    print('list_annotation_files', resp.status_code, resp.json())

    resp = client.post('/api/load_annotations', json={'file_path':'annotations_test/annotations.json'})
    print('load_annotations', resp.status_code, resp.json())

    resp = client.get('/api/categories')
    print('categories', resp.status_code, resp.json())

    resp = client.get('/api/images')
    print('images', resp.status_code, resp.json())

    # if images present, request thumbnail by id
    imgs = client.get('/api/images').json().get('items', [])
    if imgs:
        first_id = imgs[0]['image_id']
        t = client.get(f'/api/thumbnail?id={first_id}&max_size=64')
        print('thumbnail status', t.status_code, 'content-type', t.headers.get('content-type'))
    else:
        print('no images to request thumbnail for')

print('Finished')

