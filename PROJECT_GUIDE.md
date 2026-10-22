# PROJECT_GUIDE.md — the whole project, explained for teammates
*Read this before the viva/demo. It covers what we built, why, the numbers,
and what to say when teachers ask. Nothing here is fluff — every claim maps
to a real file or measured number.*

## 1. The one-paragraph version (memorize this)
During disasters, responders drown in unstructured text (tweets, SMS, radio
logs). Our project turns short free-text disaster reports into a structured,
auditable operational picture on a map dashboard. A **fine-tuned transformer**
does triage classification, an **LLM (Gemini) with a strict JSON schema**
generates the action plan, and we benchmarked everything against classical
baselines. The fine-tuned DistilBERT hits **77.8% accuracy on disasters it
never trained on**, beating both the TF-IDF baseline and a zero-shot LLM.

## 2. System architecture (draw this on the board if asked)

```
Free-text report (tweet / SMS / manual input)
        |
        v
+----------------+     +------------------+
| Fine-tuned     |     | Location         |
| DistilBERT     |     | (from Gemini     |
| category +     |     |  JSON: lat/lng)  |
| confidence     |     |                  |
+-------+--------+     +--------+---------+
        |                       |
        +----------+------------+
                   v
        Corroboration idea (design only, not validated)
                   v
        Gemini planner (strict JSON schema)
        gets triage label as context
                   v
        Dashboard: map pins, resources, NDMA phases, sitrep, contacts
```

**Design principle (say this):** the LLM does structured extraction and
planning, but it is *not the sole decision-maker* — a deterministic,
measurable classifier handles triage. That's our answer to "LLMs hallucinate."

## 3. Repo map — where everything lives

| Path | What it is |
|---|---|
| `livecrisis_dashboard/index.html` | The dashboard (Bootstrap dark + Leaflet map). We added 5 lines: banner shows `LOCAL TRIAGE: label (conf%)` |
| `livecrisis_dashboard/backend/app.py` | Flask API. `POST /analyze` (Gemini pipeline), `POST /classify` (local model, added by us). Triage label is injected into the Gemini prompt as context |
| `src/data.py` | Downloads HumAID, pools 76,484 tweets, splits **by event**, saves parquet + split IDs |
| `src/baseline_tfidf.py` | TF-IDF + Logistic Regression baseline |
| `src/train_distilbert.py` | DistilBERT fine-tune (fp16, class weights, OOM auto-fallback). `src/train_bert.py` reuses it for BERT-base |
| `src/evaluate_models.py` | Curves, confusion matrices, error analysis, comparison table |
| `src/classify.py` | `classify(text)` used by the app; returns `None` gracefully if model files are missing |
| `src/deepseek_zeroshot.py` / `src/gemini_zeroshot.py` | Zero-shot LLM baselines (DeepSeek completed; Gemini quota-blocked, 19 samples kept as preliminary) |
| `models/distilbert/`, `models/bert_base/` | Trained weights + tokenizer + label map (loadable offline) |
| `results/` | metrics.json, comparison table, 3 confusion matrices, training curves, error analyses, environment.txt |
| `configs/config.yaml` | All hyperparameters + split definition (single source of truth) |
| `run_all.sh` | Reproduces the entire pipeline end to end |
| `NOTES.md`, `STATUS.md`, `RESULTS.md` | Inspection notes, progress log, full results write-up |

## 4. Data (know this cold)
- **Dataset:** HumAID (QCRI) — ~77K human-labeled tweets from 19 real disasters
  (2016–2019): hurricanes, earthquakes, floods, wildfires. Public, from Hugging Face.
- **Labels (10, not 11):** `dont_know_cant_judge` appears in *zero* events, so the
  task is 10-class. Biggest class: rescue_volunteering (~21K). Smallest:
  missing_or_found_people (358 total). This imbalance is why accuracy alone
  lies and **macro-F1 is our primary metric**.
- **Split by EVENT (our strongest methodological point):** train = 13 events
  (52,911), val = 3 events (10,701), test = 3 *completely unseen* events —
  California wildfires, Cyclone Idai, Ecuador quake (12,872). Seed 42, saved
  to `data/splits/`. Val has 9/10 classes (rarest class kept in train), so
  checkpoint selection uses macro-F1 over val classes present.

## 5. Models & results (the numbers table — memorize the bold row)

| Model | Test acc | Macro-F1 | Note |
|---|---|---|---|
| TF-IDF + LogReg | 0.720 | 0.635 | Baseline, minutes on CPU |
| **DistilBERT (ours, deployed)** | **0.778** | **0.698** | 4 epochs, fp16, ~7 ms/tweet |
| BERT-base | 0.784 | 0.706 | Only +0.6pp at 2x latency — confirms DistilBERT was the right pick |
| DeepSeek-V4.1 zero-shot (n=30) | 0.700 | 0.415 | High accuracy, collapses on rare classes — *this* is why fine-tuning wins |
| Gemini zero-shot | prelim only (n=19, quota-blocked) | — | Honestly reported as incomplete |

Training setup: lr 2e-5, max_len 128, effective batch 32 (16×2 grad accumulation;
halve on OOM), class-weighted loss, best checkpoint by val macro-F1.

## 6. How to demo it (2 minutes)
1. `cd livecrisis_dashboard/backend && python app.py` (needs `.env`, already present)
2. Open `index.html`, type e.g. *"Flood in Assam, 3 districts affected, 50000 displaced"*
3. Point at: severity banner (now with `LOCAL TRIAGE` label), map pins, resources,
   NDMA phases, sitrep, contacts.
4. Say: "The triage label comes from our local fine-tuned model running on the
   GPU in ~7 ms; Gemini only writes the action plan around it."

## 7. Expected Q&A (teacher questions → what to say)

**Q: Why not just use an LLM for everything?**
A: Three measured reasons: (1) our zero-shot LLM gets 0.415 macro-F1 vs 0.698
fine-tuned — it ignores rare but critical classes; (2) LLMs can fabricate
locations/counts, so ours is constrained to a strict JSON schema; (3) the
local classifier runs offline in milliseconds with no API cost or quota.

**Q: Why split by event instead of randomly?**
A: Random splits leak — tweets from the same disaster appear in train and test,
inflating scores. Event split tests *generalization to unseen disasters*,
which is the real deployment scenario. It's the hardest fair test.

**Q: Why macro-F1 and not accuracy?**
A: 21K vs 358 class imbalance. A model predicting only the majority class
scores high accuracy while being useless for rare emergencies. Macro-F1
forces equal weight per class.

**Q: BERT-base barely beats DistilBERT — was it worth it?**
A: Yes, as a *ceiling check*, not as the product. It proves DistilBERT
captures nearly all achievable performance at half the latency — that's a
result, not a failure. We report it honestly.

**Q: Your Gemini baseline is incomplete. Isn't that a hole?**
A: It's disclosed as preliminary (free-tier quota: 5 req/min, 20/day). The
DeepSeek zero-shot (n=30, complete) carries the RQ1 comparison instead, and
the substitution is documented in metrics and RESULTS.md. No numbers invented.

**Q: Does it work offline / in a real disaster?**
A: Partially — and we say so. The classifier runs fully offline; the planner
needs the Gemini API. No disconnected-network test was done (listed limitation).
SMS/radio ingestion and mesh input are future scope.

**Q: What would you do with more time?**
A: Multilingual + code-mixed support (MuRIL/IndicBERT — our data is English-only),
full 200-sample Gemini baseline, NER + geocoding for locations instead of
LLM-extracted coordinates, and field validation of the corroboration heuristic.

**Q: What broke during the project?**
A: Good question — be honest: (1) first training run crashed at epoch-1 eval
from a loss-function return-type bug, fixed and verified with a smoke test;
(2) laptop sleep stalled a run (now: plugged in + sleep off);
(3) RTX 5060 needed a CUDA 12.8 torch build (installed 2.11.0+cu128).
All logged in STATUS.md.

## 8. Glossary (if a term comes up)
- **Fine-tuning:** taking a pre-trained transformer and training it a few more
  epochs on our labels. **Zero-shot:** asking a model with no task training.
- **fp16 / mixed precision:** half-precision math — ~2x faster training on GPU.
- **Gradient accumulation:** simulating a big batch with small GPU-friendly steps.
- **Checkpoint selection:** saving the model each epoch, keeping the one with
  best val macro-F1.
- **Class weights:** penalizing mistakes on rare classes more, to fight imbalance.
- **Confusion matrix:** grid of true-vs-predicted labels showing *which* classes
  get confused. **Error analysis:** sampled mistakes with hypothesized causes.
