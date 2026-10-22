"""Step 4 smoke test: 5 sample reports through /classify (Flask test client)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / 'livecrisis_dashboard' / 'backend'))
from app import app  # noqa

SAMPLES = [
    'Flood in Assam, 3 districts affected, 50000 displaced, need boats',
    'Magnitude 6.1 earthquake near Guwahati, buildings collapsed, casualties feared',
    'Cyclone approaching Odisha coast, evacuation of coastal villages underway',
    'Happy to donate blankets and food for Kerala flood victims, contact me',
    'Beautiful sunny day at the beach, nothing to report',
]

client = app.test_client()
ok = 0
for s in SAMPLES:
    r = client.post('/classify', json={'text': s})
    body = r.get_json()
    print(f'{r.status_code} | {body} | {s[:60]}', flush=True)
    ok += r.status_code == 200
print(f'{ok}/5 classified (503 = model files missing, app still runs)', flush=True)
assert ok == 5, 'integration test failed'
print('CLASSIFY_TEST_OK', flush=True)
