import os, sys, time
sys.path.insert(0, r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor')
os.environ['IMAGE_VOLUME_PATH'] = r'C:\Users\holge\CASML4SE\repos\agentic_photography_instructor'
from webapps.annotations_browser.app import app
from fastapi.testclient import TestClient

print('Starting cancel flow test')
with TestClient(app) as client:
    # start load
    resp = client.post('/api/load_annotations', json={'file_path':'annotations_test/annotations.json'})
    print('start load:', resp.status_code, resp.json())
    # poll a few times to see progress
    for i in range(5):
        st = client.get('/api/load_status').json()
        print('status before cancel:', st)
        if st.get('state') != 'loading':
            break
        time.sleep(0.2)
    # request cancel
    c = client.post('/api/load_cancel')
    print('cancel request:', c.status_code, c.json() if c.status_code==200 else c.text)
    # poll until cancelled or error/done
    for i in range(30):
        st = client.get('/api/load_status').json()
        print('status after cancel attempt:', st)
        if st.get('state') in ('cancelled','done','error'):
            break
        time.sleep(0.2)
    print('final status:', client.get('/api/load_status').json())

print('Test finished')

