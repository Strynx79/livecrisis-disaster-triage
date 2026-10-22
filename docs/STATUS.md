# STATUS.md — progress log

## Step 0 — Inspect (done 2026-10-22)
- Read `app.py`, `index.html` (head), `IDEA.md`, READMEs, `requirements.txt`, `test.py`.
- Wrote `NOTES.md` (architecture + JSON schema + integration preview).
- Verified GPU: RTX 5060 Laptop GPU, torch 2.11.0+cu128, CUDA matmul OK.
- Wrote `results/environment.txt`.
- Pre-existing issue noted, not touched: `backend/test.py` has an unresolved
  merge-conflict block (both sides identical).
- Next: Step 1 (data) per approved plan.

## Step 1 — Data (done 2026-10-22)
- Installed: transformers 5.20.2, datasets 5.1.0, sklearn 1.9.1, numpy 2.5.4,
  accelerate 1.15.0 (+pandas, matplotlib, seaborn). environment.txt updated.
- Dataset: QCRI/HumAID-events (HF Hub, verified) — pooled all 19 events x all
  splits = 76,484 tweets, 0 empties/dups. 10 classes (dont_know_cant_judge
  absent everywhere). Saved data/humaid_pooled.parquet.
- Split BY EVENT (seed 42, ids in data/splits/):
  train 52,911 (13 events) / val 10,701 (florence, kaikoura, canada) /
  test 12,872 (california, idai, ecuador). Summary: data/splits/split_summary.json.
- Imbalance is severe: rescue_volunteering ~21k vs missing_or_found_people 358
  (train keeps maryland+greece for it; val has 9/10 classes — selection =
  macro-F1 over val classes present, rarest class test-only). Class weights at train.
## Step 2 — Models (overnight run launched ~02:45 IST 2026-10-22)
- Baseline TF-IDF+LR done: val acc 0.7039 / macro-F1 0.6748; TEST acc 0.7202 /
  macro-F1 0.6352. Saved to results/metrics.json + comparison_table.csv +
  confusion_matrix_tfidf_lr.png. (Latency measured in batch; honest per-example
  number comes from DistilBERT single-inference timing.)
- DistilBERT training launched (persistent bg proc): 4 epochs, lr 2e-5,
  max_len 128, fp16, eff-batch 32 (16x2, auto-fallback 8x4 on OOM), class
  weights, best-by-val-macroF1 -> models/distilbert/. Eval script chained after.
## Step 2 — Models, cont. (overnight run FAILED at epoch-1 eval, bug fixed AM)
- Cause (mine, in src/train_distilbert.py): custom compute_loss returned a bare
  loss tensor; Trainer's train loop tolerates that but the EVAL loop unpacks
  strictly -> "TypeError: iteration over a 0-d tensor" at the first epoch eval.
  Fixed: return (loss, outputs) when return_outputs=True; read labels without
  popping (prediction_step reads inputs['labels'] after the call).
- Verified with scripts/smoke_eval.py: trainer.evaluate() on 64 ex -> SMOKE_EVAL_OK.
  WeightedTrainer hoisted to module level for reuse.
- Secondary findings: no usable checkpoint was saved (crash hit before save);
  full restart required. Observed ~4 it/s -> 4 epochs ~= 8-10h. Machine slept
  mid-run (timestamps jump 05:26 -> 6:55, step rate collapsed) — relaunch needs
  sleep disabled + plugged in.
- Baseline stands: test acc 0.7202 / macro-F1 0.6352 (target to beat).
- Awaiting user decision: relaunch tonight keep-awake, or run now in background.
- Relaunch APPROVED, started morning 2026-10-22 (persistent bg proc, same chain).
  Tip logged for user: keep plugged in, disable sleep for fastest finish.
## Step 4 — Integration (done 2026-10-22)
- src/classify.py: classify(text) -> {label, confidence} via models/distilbert
  (GPU/CPU), None when model files absent.
- app.py: new POST /classify (503 when model missing); /analyze feeds the triage
  label into the Gemini prompt as context and echoes it as result['triage'].
  All edits marked ML; zero new hard deps (lazy torch import).
- index.html: one marked addition in updateDashboard — banner appends
  "LOCAL TRIAGE: label (conf%)" when present. No restyle.
- scripts/test_classify.py via Flask test_client: 5/5, e.g. Assam flood ->
  displaced_people_and_evacuations (0.73), donation -> rescue_volunteering (0.99).
  Honest note: "sunny beach day" -> other_relevant_information (0.60), not
  not_humanitarian — model weak on that class, for RESULTS.md limitations.

## Step 3 — Gemini zero-shot (BLOCKED by free-tier quota, script hardened)
- Root cause of 181/200 invalid + 0/50 pipeline: HTTP 429, NOT model quality.
  Key's free tier = 5 req/min + 20 req/day on gemini-3.6-flash. Only 19 valid
  labels resolved before quota exhaustion (all cached, kept).
- Hardened src/gemini_zeroshot.py: 13s spacing, honors retry_delay (cap 120s),
  NEVER caches failures, reruns fill gaps (19 already banked). metrics.json
  marked PRELIMINARY (complete:false) so nothing overclaims.
- Awaiting user call: (a) trickle ~20/day to 200 (~9 more days), (b) smaller
  subset, or (c) different model/key with bigger quota.
- User chose (b): subset shrunk to 30 label + 10 pipe (seed-42 draw keeps the
  19 banked). Quota still red at patch time, so cron job 01df857476e0 fires
  the resume script once in ~6h (after reset); failures never cache, so any
  remainder just needs one more run. Deliver is local-only; check by hand.

## Deliverables checklist (closed 2026-10-22, ~1.5h budget)
[x] NOTES.md (architecture + JSON schema)
[x] results/comparison_table.* + metrics.json (tfidf/distilbert/bert + gemini prelim)
[x] confusion matrices (3), training curves (tagged), error_analysis (tagged)
[x] models/{distilbert,bert_base}/ + working /classify integration (5/5 test)
[x] run_all.sh (SH_OK), README section, requirements-ml.txt, environment.txt
[x] RESULTS.md (exhaustive; Gemini marked PRELIMINARY/complete:false)
[x] Zero-shot LLM baseline via DeepSeek (Gemini track superseded by quota; see Step 5)
[ ] test.py merge-conflict cleanup (pre-existing, out of scope)
## Step 5 — DeepSeek zero-shot (DONE via OpenCode go-plan key, 2026-10-22)
- deepseek-v4.1-flash is real (Zen catalog) at zen/go/v1/chat/completions.
  Issues hit: 402 no funds (fixed by go plan), 400 MissingSessionID (fixed with
  x-opencode-session header). Key lives only in git-ignored backend/.env.
- 30/30 clean: test-subset acc 0.7000 / macro-F1 0.4152 / weighted 0.7378.
  Added to metrics.json + comparison table. Gemini 19-sample data kept separate.

## BERT-base (DONE — marginal gain, honest story)
- Train+eval exited 0. Batch 8x4, no OOM, 3 epochs, best val macro-F1 0.7316.
- TEST: acc 0.7839 / macro-F1 0.7064 / 14.2ms single-ex. vs DistilBERT 0.7781 /
  0.6976 / 6.8ms: +0.6pp accuracy at ~2x latency. Paper line: DistilBERT is the
  practical choice; BERT-base confirms the benchmark ceiling.
- All tagged artifacts saved (curves, confusion, error_analysis_bert_base.md).
- Remaining after all numbers land: run_all.sh + RESULTS.md.

## Step 2 — Models (DISTILBERT DONE, beats baseline)
- Train+eval chain exited 0. Batch 16x2, no OOM, 4 epochs.
- Best val macro-F1 0.6526 -> models/distilbert/ (weights+tokenizer+label_map).
- TEST (12,872, unseen events): acc 0.7781 / macro-F1 0.6976 / single-ex latency
  6.8ms. Baseline was 0.7202 / 0.6352. All artifacts: curves, both confusion
  matrices, error_analysis.md (2,856 errors), comparison_table, metrics.json.
  (Section order: this completed before Steps 3/4 above.)
