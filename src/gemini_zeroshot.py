"""Gemini zero-shot baseline: 200-sample test subset + 50-sample pipeline check.

- Label eval: classify into the 10 HumAID labels, exact-label-only replies.
- Pipeline eval: POST /analyze via Flask test_client, check HTTP 200 + schema keys.
- Retries (3, backoff), disk cache (data/gemini_cache.json), resume-safe.
- No GPU needed; safe to run alongside BERT training.
- Free-tier quota aware: 13s spacing (5 req/min), honors retry_delay (capped),
  NEVER caches failures -> resume-safe. Rerun anytime; it fills in the gaps.
"""
import json
import random
import re
import time
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA, SPLITS, RESULTS = ROOT / 'data', ROOT / 'data' / 'splits', ROOT / 'results'
CACHE = DATA / 'gemini_cache.json'
SEED, N_LABEL, N_PIPE = 42, 30, 10   # shrunk 2026-10-22: free tier is 20/day;
# the fixed seed-42 draw keeps the 19 already-banked labels inside the subset.
MODEL = 'gemini-3.6-flash'
SPACING_S = 13  # free tier: 5 requests/min
SCHEMA_KEYS = ['incident_location', 'severity_title', 'severity_level',
               'population_affected', 'estimated_response_time', 'resources',
               'ndma_phases', 'situation_report', 'emergency_contacts', 'map_pins']


def load_cache():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save_cache(c):
    CACHE.write_text(json.dumps(c))


def call_gemini(model, prompt, tries=3):
    """Returns (text, None) on success, (None, err) on persistent failure."""
    last = 'unknown'
    for a in range(tries):
        try:
            return model.generate_content(prompt).text, None
        except Exception as e:
            last = str(e)
            m = re.search(r'retry in ([\d.]+)s', last)
            wait = min(float(m.group(1)) if m else 2 * (a + 1), 120)
            print(f'429/backoff {wait:.0f}s (try {a + 1}/{tries})', flush=True)
            time.sleep(wait)
    return None, last


def norm(s):
    return s.strip().lower().replace(' ', '_').replace('-', '_')


def main():
    import google.generativeai as genai  # noqa
    from dotenv import load_dotenv
    import os
    load_dotenv(ROOT / 'livecrisis_dashboard' / 'backend' / '.env')
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel(MODEL)
    summary = json.loads((SPLITS / 'split_summary.json').read_text())
    labels = summary['labels']

    test_ids = json.loads((SPLITS / 'test_ids.json').read_text())
    random.seed(SEED)
    sub = random.sample(test_ids, N_LABEL)
    (RESULTS / 'gemini_subset_ids.json').write_text(json.dumps(sub))
    pooled = pd.read_parquet(DATA / 'humaid_pooled.parquet')
    tmap = {str(i): (t, c) for i, t, c in
            zip(pooled['tweet_id'].astype(str), pooled['text'], pooled['class_label'])}
    cache = load_cache()

    y_true, y_pred, n_invalid, n_failed = [], [], 0, 0
    for i, tid in enumerate(sub):
        text, true = tmap[tid]
        key = f'label:{tid}'
        if key not in cache:
            resp, err = call_gemini(
                model, 'You are a disaster-response classifier. Reply with EXACTLY ONE '
                f'of these labels and nothing else:\n{", ".join(labels)}\n\nTweet: {text}')
            if err is None:
                cache[key] = resp
                save_cache(cache)
            else:
                n_failed += 1
                print(f'{tid}: FAILED, will retry next run', flush=True)
            time.sleep(SPACING_S)
            if err is not None:
                continue
        pred = norm(cache[key])
        if pred not in labels:
            n_invalid += 1
            print(f'{tid}: unparseable reply {cache[key][:80]!r}', flush=True)
            continue
        y_true.append(true)
        y_pred.append(pred)
        if (i + 1) % 50 == 0:
            print(f'label eval {i + 1}/{N_LABEL} (resolved {len(y_pred)})', flush=True)
    complete = (n_failed == 0)
    valid_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    valid_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) if y_true else 0.0
    print(f'label: valid={len(y_pred)}/{N_LABEL} invalid={n_invalid} failed={n_failed} '
          f'complete={complete} valid_acc={valid_acc:.4f} valid_macroF1={valid_macro:.4f}',
          flush=True)

    # --- pipeline JSON-validity on 50-sample subset via real /analyze route ---
    sys_path = str(ROOT / 'livecrisis_dashboard' / 'backend')
    import sys
    sys.path.insert(0, sys_path)
    from app import app as flask_app  # noqa
    pipe_ids = random.sample(test_ids, N_PIPE)
    client = flask_app.test_client()
    ok200, schema_ok, pipe_failed = 0, 0, 0
    for i, tid in enumerate(pipe_ids):
        text, _ = tmap[tid]
        key = f'pipe:{tid}'
        if key not in cache:
            r = client.post('/analyze', json={'scenario': text, 'disaster_type': ''})
            try:
                body = r.get_json()
            except Exception:
                body = None
            if r.status_code == 200 and isinstance(body, dict):
                cache[key] = {'status': r.status_code, 'body': body}
                save_cache(cache)
            else:
                pipe_failed += 1
                print(f'pipe {tid}: status={r.status_code}, retry next run', flush=True)
            time.sleep(SPACING_S)
            if not (r.status_code == 200 and isinstance(body, dict)):
                continue
        ent = cache[key]
        if ent['status'] == 200 and isinstance(ent['body'], dict):
            ok200 += 1
            if all(k in ent['body'] for k in SCHEMA_KEYS):
                schema_ok += 1
        if (i + 1) % 25 == 0:
            print(f'pipeline eval {i + 1}/{N_PIPE}', flush=True)
    print(f'pipeline: http200={ok200}/{N_PIPE} schema_complete={schema_ok}/{N_PIPE} '
          f'failed={pipe_failed}', flush=True)

    mp = RESULTS / 'metrics.json'
    prev = json.loads(mp.read_text()) if mp.exists() else {}
    prev['gemini_zeroshot'] = {
        'model': MODEL, 'n': N_LABEL, 'valid_responses': len(y_pred),
        'invalid_responses': n_invalid, 'failed_quota': n_failed,
        'complete': complete, 'valid_accuracy': valid_acc,
        'valid_macro_f1': valid_macro}
    prev['gemini_pipeline'] = {
        'n': N_PIPE, 'http_200': ok200, 'schema_complete': schema_ok,
        'failed_quota': pipe_failed, 'json_valid_rate': ok200 / N_PIPE}
    mp.write_text(json.dumps(prev, indent=1))
    print('saved gemini metrics', flush=True)


if __name__ == '__main__':
    main()
