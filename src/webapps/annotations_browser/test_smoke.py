from webapps.annotations_browser.app import app
from fastapi.testclient import TestClient

with TestClient(app) as client:
    r = client.post('/api/reload', json={})
    print('POST /api/reload', r.status_code)
    try:
        print(r.json())
    except Exception as e:
        print('No JSON body or error:', e)

    r = client.get('/api/categories')
    print('GET /api/categories', r.status_code)
    try:
        print(r.json())
    except Exception as e:
        print('No JSON body or error:', e)

    r = client.get('/api/images')
    print('GET /api/images', r.status_code)
    try:
        print(r.json())
    except Exception as e:
        print('No JSON body or error:', e)

