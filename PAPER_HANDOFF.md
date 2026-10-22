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

---

## DATA APPENDIX (everything below is measured; paste the whole file as context)

### A. Comparison table (test set, n=12,872 unseen-event tweets, unless noted)
| model | split | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|---|
| TF-IDF+LR | test | 0.7202 | 0.6352 | 0.7238 |
| TF-IDF+LR | val | 0.7039 | 0.6748 | (in metrics.json) |
| DistilBERT | test | 0.7781 | 0.6976 | 0.7751 |
| BERT-base | test | 0.7839 | 0.7064 | 0.7811 |
| DeepSeek-V4.1-Flash zero-shot | test-subset (n=30) | 0.7000 | 0.4152 | 0.7378 |
| Gemini-3.6-flash zero-shot | PRELIMINARY (n=19) | 0.7368 | 0.6510 | — (NOT comparable) |
Latencies: DistilBERT 6.8 ms/ex single, BERT 14.2 ms/ex single (RTX 5060).
Val selection scores: DistilBERT best val macro-F1 0.6526; BERT-base 0.7316.

### B. Per-class test F1 (precision / recall / F1)
TF-IDF+LR: caution 0.487; displaced 0.723; infrastructure 0.703; injured 0.907;
missing_or_found 0.510; not_humanitarian 0.615; other_relevant 0.405;
requests_urgent 0.444; rescue_volunteering 0.796; sympathy 0.762.
DistilBERT: caution P0.550 R0.597 F1 0.572; displaced 0.771/0.816/0.793;
infrastructure 0.752/0.803/0.776; injured 0.901/0.960/0.929; missing_or_found
0.833/0.505/0.629; not_humanitarian 0.628/0.744/0.681; other_relevant
0.559/0.377/0.450; requests_urgent 0.378/0.671/0.484; rescue_volunteering
0.902/0.818/0.858; sympathy 0.768/0.841/0.803.
BERT-base: caution 0.561/0.609/0.584; displaced 0.734/0.852/0.789;
infrastructure 0.775/0.778/0.776; injured 0.901/0.965/0.932; missing_or_found
0.904/0.520/0.660; not_humanitarian 0.643/0.752/0.693; other_relevant
0.554/0.393/0.460; requests_urgent 0.389/0.668/0.492; rescue_volunteering
0.899/0.826/0.861; sympathy 0.800/0.837/0.818.
Reading: all models ace injured_or_dead_people (~0.93) and struggle with
other_relevant_information (~0.45) and requests_or_urgent_needs (~0.49) —
vague, overlapping categories. The zero-shot LLM collapses on rare classes;
fine-tuned models hold precision on missing_or_found_people (0.83–0.90)
despite only 358 training examples.

### C. Split + training config (configs/config.yaml, seed 42)
Train (52,911, 13 events): greece_wildfires, dorian, harvey, irma, maria,
matthew_2016, italy_earthquake, kerala_floods, maryland_floods,
midwestern_floods, pakistan_earthquake, puebla_earthquake, srilanka_floods.
Val (10,701): florence, kaikoura_earthquake, canada_wildfires (9/10 classes).
Test (12,872): california_wildfires, cyclone_idai, ecuador_earthquake.
DistilBERT: distilbert-base-uncased, 4 epochs, lr 2e-5, max_len 128, fp16,
eff-batch 32 (16x2), class weights, best-by-val-macroF1. BERT-base: 3 epochs,
8x4. TF-IDF: 1–2 grams, 20k features, C=1.0, balanced weights.

### D. Environment (results/environment.txt)
RTX 5060 Laptop 8GB, driver 617.14, torch 2.11.0+cu128, CUDA 12.8, python
3.12.10, transformers 5.20.2, datasets 5.1.0, sklearn 1.9.1, accelerate 1.15.0,
flask 3.1.3, openai 3.36.1. Planner: DeepSeek-V4.1-Flash (OpenCode Zen go endpoint).

### E. File manifest (repo root: github.com/Strynx79/livecrisis-disaster-triage)
Code: src/{data,baseline_tfidf,train_distilbert,train_bert,evaluate_models,
classify,deepseek_zeroshot,gemini_zeroshot}.py, scripts/test_classify.py.
Models: models/{distilbert,bert_base}/ (weights EXCLUDED from git >100MB;
tokenizer+config+label_map included; retrain via run_all.sh). Results:
metrics.json, comparison_table, 3 confusion matrices, 2 training-curve plots,
2 error analyses (2,856 / 2,781 test errors, 15 sampled each), environment.txt.
Docs: docs/{IDEA,NOTES,PROJECT_GUIDE,README,STATUS}.md, RESULTS.md,
livecrisis-architecture.html, configs/config.yaml, requirements-ml.txt.
