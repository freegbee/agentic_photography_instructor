import os, sys
sys.path.insert(0, r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor')
from src.webapps.annotations_browser.app import app
print('App imported')
print('Has annotations_index?', hasattr(app.state, 'annotations_index'))
print('annotations_index_summary:', getattr(app.state, 'annotations_index_summary', None))

# TestClient calls
from fastapi.testclient import TestClient
with TestClient(app) as c:
    print('Calling POST /api/reload')
    r = c.post('/api/reload', json={})
    print('POST status', r.status_code)
    try:
        print('POST body:', r.json())
    except Exception as e:
        print('POST body error', e)
    r = c.get('/api/categories')
    print('GET /api/categories status', r.status_code)
    try:
        print('GET /api/categories body', r.json())
    except Exception as e:
        print('GET categories error', e)
    r = c.get('/api/images')
    print('GET /api/images status', r.status_code)
    try:
        print('GET /api/images body', r.json())
    except Exception as e:
        print('GET images error', e)

