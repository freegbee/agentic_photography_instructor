"""Simple smoke test for the metrics endpoint."""
import sys
import requests

URL = 'http://localhost:8000/metrics'

try:
    r = requests.get(URL, timeout=5)
    if r.status_code == 200 and 'process_cpu_seconds_total' in r.text:
        print('OK')
        sys.exit(0)
    else:
        print('Metrics endpoint returned unexpected content or status:', r.status_code)
        sys.exit(2)
except Exception as e:
    print('Failed to fetch metrics:', e)
    sys.exit(1)

