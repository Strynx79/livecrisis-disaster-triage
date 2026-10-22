# PAPER_HANDOFF.md — prompt + verification kit for Claude (IEEE paper, due tomorrow AM)

## Copy everything below the line into Claude. Then run the checklist yourself.

---

You are writing an IEEE-format conference paper for an undergraduate deep-learning
course project (5% of grade, conference track). All facts, numbers, and artifacts
below are measured except where explicitly marked PRELIMINARY/UNMEASURED — never
present the marked ones as results, never invent numbers, never round favorably.

### Project in one paragraph
LiveCrisis: a channel-agnostic disaster-triage pipeline. Free-text reports go to a
fine-tuned DistilBERT classifier (category + confidence) and a strict-JSON LLM
planner, rendered on a Leaflet dashboard (map, resources, NDMA phases, sitrep,
contacts). Benchmarked on HumAID crisis tweets with an event-based split.

### Required structure (IEEE)
1. Abstract 2. Introduction 3. Related work (crisis informatics, BERT/DistilBERT,
HumAID/CrisisNLP, LLM structured extraction, Ushahidi/AIDR/Sahana) 4. Methodology
(pipeline, 10-class label set, event split, training setup) 5. Experiments
6. Results (table + confusion matrices + error analysis) 7. Discussion (hybrid
trade-offs, channel reliability, limitations) 8. Conclusion. Citation rule: only
cite references you can verify exist; drop anything uncertain.

### Research questions
RQ1: fine-tuned transformer vs classical baseline vs zero-shot LLM.
RQ2: schema-constrained LLM extraction reliability (PARTIALLY ANSWERED — see limits).
RQ3: hybrid design trustworthiness. RQ4 (analytical): offline channel reliability.

### Citable numbers (hardware: RTX 5060 Laptop 8GB, torch 2.11.0+cu128, seed 42)
- Data: 76,484 tweets, 19 events, 10 classes; train 52,911 (13 events) /
  val 10,701 (3) / test 12,872 (3 fully unseen: California wildfires, Idai, Ecuador).
- TF-IDF+LR test: acc 0.7202, macro-F1 0.6352, weighted 0.7238.
- DistilBERT (4ep, best val 0.6526) test: acc 0.7781, macro-F1 0.6976,
  weighted 0.7751, ~6.8 ms/example. THE DEPLOYED MODEL.
- BERT-base (3ep, best val 0.7316) test: acc 0.7839, macro-F1 0.7064,
  weighted 0.7811, ~14.2 ms/example. Frame as ceiling check, not the pick.
- DeepSeek-V4.1-Flash zero-shot (n=30 independent subset): acc 0.7000,
  macro-F1 0.4152, weighted 0.7378. Headline contrast: 0.698 vs 0.415 macro-F1.
- Integration: /classify 5/5 live; graceful 503 without model files.
- Training: lr 2e-5, max_len 128, eff-batch 32 (grad accumulation), fp16,
  class weights, macro-F1 selection.

### BANNED from results (do not report as findings)
- Gemini zero-shot: only 19/200 labels resolved (free-tier quota). Mention once
  as a preliminary probe (acc 0.737/macro 0.651 on n=19, NOT comparable), never
  in the comparison table.
- Pipeline JSON-validity: 0/50 measured — ALL blocked by quota (HTTP 500s).
  Report as "not measured," never as 0%.
- `dont_know_cant_judge` (11th HumAID label): absent from all events; task is 10-class.
- Corroboration score: designed, never field-validated. Future work only.
- Val set covers 9/10 classes (rarest class train/test-only by design).

### Figures (in results/, reference by filename)
confusion_matrix_{tfidf_lr,distilbert,bert_base}.png,
training_curves_{distilbert,bert_base}.png, comparison_table.md,
error_analysis_{distilbert,bert_base}.md. Metrics: results/metrics.json.
Architecture visual: livecrisis-architecture.html. Team explainer: docs/PROJECT_GUIDE.md.

### Tone rules
Report losses honestly (BERT margin, beach-day misclassification, quota failures).
Limitations section is mandatory and must include: tweet-vs-SMS gap, English-only,
small LLM subset, no field/disconnected test, single seed. Demo ≠ evidence: the
dashboard proves integration; only the benchmark proves the claim.

---

## Your verification checklist (do this to Claude's draft, not mine)
- [ ] Every number traces to results/metrics.json or the table above — spot-check 5.
- [ ] No banned item appears in results/conclusion; Gemini only in limitations.
- [ ] No invented citations; every reference real and relevant.
- [ ] Macro-F1 is the primary metric everywhere, never accuracy-alone claims.
- [ ] Limitations section present with all 5 items; no "future work" framed as done.
- [ ] Confusion-matrix/error-analysis figures referenced in text, not just appended.
- [ ] Demo described as integration proof, benchmark as evidence — never swapped.
