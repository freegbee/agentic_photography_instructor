import os, sys, time, json
sys.path.insert(0, r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor')
os.environ['IMAGE_VOLUME_PATH'] = r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor'
from webapps.annotations_browser.app import app
from fastapi.testclient import TestClient

with TestClient(app) as client:
    print('list files:', client.get('/api/list_annotation_files').json())
    resp = client.post('/api/load_annotations', json={'file_path':'annotations_test/annotations.json'})
    print('start load resp:', resp.status_code, resp.json())
    # poll status
    for i in range(60):
        st = client.get('/api/load_status').json()
        print('status:', st)
        if st.get('state') in ('done','error'):
            break
        time.sleep(0.2)
    print('final status:', client.get('/api/load_status').json())
    print('images:', client.get('/api/images').json())

print('done')

