"""DeepSeek-V4-Flash zero-shot baseline (OpenCode Zen, OpenAI-compatible).

Same 10 HumAID labels, same prompt shape as the Gemini probe, but a FRESH
fixed 30-sample subset (first 30 of results/gemini_subset_ids.json) evaluated
entirely on ONE model: deepseek-v4-flash. The 19 Gemini answers stay separate
preliminary data — never mixed into this metric.
API key read from backend/.env (OPENCODE_API_KEY), never logged.
"""
import json
import time
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA, SPLITS, RESULTS = ROOT / 'data', ROOT / 'data' / 'splits', ROOT / 'results'
CACHE = DATA / 'deepseek_cache.json'
MODEL = 'deepseek-v4.1-flash'
BASE_URL = 'https://opencode.ai/zen/go/v1'  # go-plan endpoint (2026-10-22)
N = 30
SPACING_S = 5


def load_dotenv_min(path):
    d = {}
    for line in Path(path).read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            d[k.strip()] = v.strip()
    return d


def call_llm(client, prompt, tries=3):
    import uuid
    last = 'unknown'
    session = str(uuid.uuid4())
    for a in range(tries):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                extra_headers={'x-opencode-session': session},
                messages=[{'role': 'user', 'content': prompt}])
            return r.choices[0].message.content, None
        except Exception as e:
            last = str(e)[:200]
            print(f'API error (try {a + 1}/{tries}): {last}', flush=True)
            time.sleep(min(10 * (a + 1), 60))
    return None, last


def norm(s):
    return s.strip().lower().replace(' ', '_').replace('-', '_')


def main():
    from openai import OpenAI  # noqa
    env = load_dotenv_min(ROOT / 'livecrisis_dashboard' / 'backend' / '.env')
    client = OpenAI(base_url=BASE_URL, api_key=env['OPENCODE_API_KEY'])
    summary = json.loads((SPLITS / 'split_summary.json').read_text())
    labels = summary['labels']

    full_sub = json.loads((RESULTS / 'gemini_subset_ids.json').read_text())
    sub = full_sub[:N]
    (RESULTS / 'deepseek_subset_ids.json').write_text(json.dumps(sub))
    pooled = pd.read_parquet(DATA / 'humaid_pooled.parquet')
    tmap = {str(i): (t, c) for i, t, c in
            zip(pooled['tweet_id'].astype(str), pooled['text'], pooled['class_label'])}
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    y_true, y_pred, n_invalid, n_failed = [], [], 0, 0
    for tid in sub:
        text, true = tmap[tid]
        key = f'label:{tid}'
        if key not in cache:
            resp, err = call_llm(
                client, 'You are a disaster-response classifier. Reply with EXACTLY ONE '
                f'of these labels and nothing else:\n{", ".join(labels)}\n\nTweet: {text}')
            if err is None:
                cache[key] = resp
                CACHE.write_text(json.dumps(cache))
            else:
                n_failed += 1
            time.sleep(SPACING_S)
            if err is not None:
                continue
        pred = norm(cache[key])
        if pred not in labels:
            n_invalid += 1
            print(f'{tid}: unparseable {cache[key][:80]!r}', flush=True)
            continue
        y_true.append(true)
        y_pred.append(pred)
    complete = (n_failed == 0 and len(y_pred) + n_invalid == N)
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0) if y_true else 0.0
    w = f1_score(y_true, y_pred, average='weighted', zero_division=0) if y_true else 0.0
    print(f'deepseek: valid={len(y_pred)}/{N} invalid={n_invalid} failed={n_failed} '
          f'acc={acc:.4f} macroF1={macro:.4f} weightedF1={w:.4f}', flush=True)

    mp = RESULTS / 'metrics.json'
    prev = json.loads(mp.read_text()) if mp.exists() else {}
    prev['deepseek_zeroshot'] = {
        'model': MODEL, 'via': 'opencode-zen', 'n': N,
        'valid_responses': len(y_pred), 'invalid_responses': n_invalid,
        'failed': n_failed, 'complete': complete, 'accuracy': acc,
        'macro_f1': macro, 'weighted_f1': w,
        'note': 'Same label set/prompt shape as Gemini probe; INDEPENDENT 30-sample '
                'subset, all on deepseek-v4-flash. Gemini 19-sample data kept separate.'}
    mp.write_text(json.dumps(prev, indent=1))
    print('saved deepseek metrics', flush=True)


if __name__ == '__main__':
    main()
