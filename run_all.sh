#!/bin/bash
# Reproduce everything: data -> baselines -> training -> evals -> integration test.
# GPU steps (train_*) need CUDA torch; Gemini step needs backend/.env with
# GEMINI_API_KEY and free-tier quota (13s spacing built in; reruns fill gaps).
set -e
cd "$(dirname "$0")"
export PYTHONPATH=
PY=.venv/Scripts/python.exe

echo '=== [0] deps (already in .venv; rejoins if missing) ==='
uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --python $PY
uv pip install transformers datasets scikit-learn numpy pandas matplotlib seaborn accelerate \
  google-generativeai flask flask-cors python-dotenv --python $PY

echo '=== [1] data ==='
$PY -u src/data.py | tail -5

echo '=== [2] TF-IDF baseline ==='
$PY -u src/baseline_tfidf.py | tail -4

echo '=== [3] DistilBERT (hours, persists through restarts) ==='
$PY -u src/train_distilbert.py | tail -4
$PY -u src/evaluate.py | tail -6

echo '=== [4] BERT-base (conditional, ~15h) ==='
$PY -u src/train_bert.py | tail -4
$PY -u src/evaluate_models.py | tail -8

echo '=== [5] Gemini zero-shot (needs key + quota) ==='
if [ -f livecrisis_dashboard/backend/.env ]; then
  $PY -u src/gemini_zeroshot.py | tail -4
else
  echo 'SKIP: no livecrisis_dashboard/backend/.env'
fi

echo '=== [6] integration test ==='
$PY -u scripts/test_classify.py | tail -7
echo 'ALL DONE: see results/, models/, STATUS.md'
