# RESULTS.md — plain-language summary (exhaustive; numbers match results/metrics.json)

## Headline
A fine-tuned DistilBERT classifies disaster tweets at **77.8% accuracy /
0.698 macro-F1 on 3 fully unseen disaster events**, beating a TF-IDF baseline
(72.0% / 0.635) by ~6 points. BERT-base adds only +0.6pp at 2x latency, so
DistilBERT is the deployed model. The Gemini zero-shot comparison is
**incomplete** (API free-tier quota) — 19/30 labels resolved, reported as
preliminary only.

## What was run
- **Data:** QCRI/HumAID-events (Hugging Face Hub), pooled all 19 events x all
  splits = **76,484 tweets, 0 empties/dups**, seed 42. Split **by event**:
  train 52,911 (13 events) / val 10,701 (florence, kaikoura, canada) /
  test 12,872 (california wildfires, idai, ecuador quake).
  `data/humaid_pooled.parquet`, ids in `data/splits/`.
- **Labels:** 10 classes (`dont_know_cant_judge` absent from all 19 events).
  Severe imbalance: rescue_volunteering ~21k vs missing_or_found_people 358.
  Mitigation: class weights + macro-F1 selection. Val has 9/10 classes
  (maryland kept in train for the rarest class) — selection is macro-F1 over
  val classes present; the rarest class is judged on test only.
- **Hardware:** RTX 5060 Laptop 8GB, driver 617.14, torch 2.11.0+cu128,
  fp16, effective batch 32 (grad accumulation). One GPU job at a time.

## Numbers (all on held-out test, n=12,872, unseen events)

| model | acc | macro-F1 | weighted-F1 | latency |
|---|---|---|---|---|
| TF-IDF+LR | 0.7202 | 0.6352 | 0.7238 | ~batch |
| DistilBERT (4ep, best val 0.6526) | **0.7781** | **0.6976** | 0.7751 | 6.8 ms/ex |
| BERT-base (3ep, best val 0.7316) | 0.7839 | 0.7064 | 0.7811 | 14.2 ms/ex |
| DeepSeek-V4.1-Flash zero-shot (n=30, OpenCode Zen) | 0.7000 | 0.4152 | 0.7378 | API |
| Gemini 3.6 flash zero-shot | (19-sample prelim: acc 0.737 / macro 0.651 — NOT comparable, tiny n) | | | |

- Zero-shot translation: DeepSeek hits 70% accuracy but only 0.415 macro-F1 —
  it leans on majority classes and largely misses rare ones. Fine-tuned
  DistilBERT (0.698 macro) beats it by 28 points where it counts: balanced
  performance across all 10 classes. That gap is the paper's core argument
  for fine-tuning over zero-shot LLMs.

- TF-IDF per-class test F1 (weakest: other_relevant_information 0.405,
  requests_or_urgent_needs 0.444, caution_and_advice 0.487; strongest:
  injured_or_dead_people 0.907). Full table: `results/comparison_table.md`.
- Gemini pipeline validity: 0/50 HTTP-200 — **all 50 blocked by quota
  (HTTP 500s), not measured**. DeepSeek (same go-plan key, OpenAI-compatible
  endpoint + session header) stands as the zero-shot baseline instead; model
  substitution documented in metrics + comparison table.
- Integration: `/classify` 5/5 sample reports correct; graceful 503 without
  model files. Honest wrinkle: "sunny beach day" -> other_relevant_information
  (0.60), not not_humanitarian.

## Artifacts (all present)
- `results/metrics.json`, `comparison_table.{csv,md}`
- `confusion_matrix_{tfidf_lr,distilbert,bert_base}.png`
- `training_curves_{distilbert,bert_base}.png`
  (legacy `training_curves.png`/`error_analysis.md` = distilbert run; tagged
  copies are canonical)
- `error_analysis_{distilbert,bert_base}.md` (2,856 / 2,781 test errors, 15 sampled each)
- `models/distilbert/`, `models/bert_base/` (weights + tokenizer + label_map)
- `configs/config.yaml`, `run_all.sh`, `requirements-ml.txt`, `results/environment.txt`

## Limitations (for the paper)
1. Tweets differ from real SMS/radio reports; English-only.
2. Event split is the strongest available design, but val covers 9/10 classes.
3. Gemini comparison incomplete (19 labels, quota-blocked pipeline check).
4. Single seed (42); no cross-validation.
5. Results depend on Gemini model version; corroboration heuristic unvalidated.
6. Not tested in a real disaster or on a disconnected network.
7. `backend/test.py` still has a pre-existing merge-conflict block (untouched).

## What failed along the way (logged, not hidden)
1. First DistilBERT run crashed at epoch-1 eval: custom loss returned a bare
   tensor (train loop tolerates, eval unpacks strictly). Fixed, verified with
   a 64-ex eval smoke test, restarted clean.
2. Laptop slept mid-run (1.5h gap); relaunch rule: plugged in + sleep off.
3. Gemini free tier (5/min, 20/day) stopped the baseline at 19/200; script now
   quota-aware (13s spacing, no failure caching) for a post-reset rerun.
