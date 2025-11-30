import os, sys
sys.path.insert(0, r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor')
# Set IMAGE_VOLUME_PATH so index resolves files relative to repo
os.environ['IMAGE_VOLUME_PATH'] = r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor'
from src.webapps.annotations_browser.app import app
from fastapi.testclient import TestClient

with TestClient(app) as c:
    print('POST /api/reload ->', c.post('/api/reload', json={'search_root': 'annotations_test'}).status_code)
    print('GET /api/categories ->', c.get('/api/categories').status_code, c.get('/api/categories').json())
    r = c.get('/api/images')
    print('GET /api/images ->', r.status_code)
    print(r.json())
    # request thumbnail for first referenced image
    t = c.get('/api/thumbnail?path=resources/sample_images/Places365_test_00000001.jpg&max_size=128')
    print('GET /api/thumbnail ->', t.status_code, t.headers.get('content-type'))

