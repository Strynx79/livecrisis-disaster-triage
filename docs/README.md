# semV_DL_project — LiveCrisis disaster-triage + DL classifier

App: `livecrisis_dashboard/` (Flask `/analyze` + `/classify`, Leaflet dashboard).
ML: `../src/`, `../configs/config.yaml`, `../models/`, `../results/`. Full story: `../RESULTS.md`.

## Training and evaluation (reproduces the paper numbers)
1. Create venv + install: `py -3.12 -m venv .venv`, then the two pip lines in
   `run_all.sh` (torch CUDA-12.8 build first — RTX 50-series needs it), or simply
   run `bash run_all.sh` end to end (data -> baseline -> DistilBERT ->
   BERT-base -> Gemini (needs `livecrisis_dashboard/backend/.env`) ->
   integration test).
2. Key scripts: `src/data.py` (HumAID pool + event split), `src/baseline_tfidf.py`,
   `src/train_distilbert.py`, `src/train_bert.py`, `src/evaluate_models.py`,
   `src/gemini_zeroshot.py`, `scripts/test_classify.py`.
3. Results land in `results/` (metrics.json, comparison table, confusion
   matrices, training curves, error analyses); models in `models/*/`.
4. Hardware reference: RTX 5060 Laptop 8 GB; DistilBERT 4 epochs ~= 8-10h at
   ~4 it/s; keep the machine plugged in with sleep disabled.
