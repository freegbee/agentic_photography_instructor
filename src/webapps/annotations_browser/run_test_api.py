import os, sys
sys.path.insert(0, r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor')
os.environ['IMAGE_VOLUME_PATH'] = r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor'
from webapps.annotations_browser.app import app
from fastapi.testclient import TestClient

print('Starting API test')
with TestClient(app) as c:
    print('Listing annotation files...')
    resp = c.get('/api/list_annotation_files')
    print('list status', resp.status_code)
    print(resp.json())

    print('Loading annotations_test/annotations.json')
    resp = c.post('/api/load_annotations', json={'file_path': 'annotations_test/annotations.json'})
    print('load status', resp.status_code)
    print(resp.json())

    print('Fetching categories...')
    resp = c.get('/api/categories')
    print('categories status', resp.status_code)
    print(resp.json())

    print('Fetching images...')
    resp = c.get('/api/images')
    print('images status', resp.status_code)
    print(resp.json())

    print('Request thumbnail...')
    resp = c.get('/api/thumbnail?path=resources/sample_images/Places365_test_00000001.jpg&max_size=64')
    print('thumb status', resp.status_code)
    print('content-type', resp.headers.get('content-type'))

print('Done')

