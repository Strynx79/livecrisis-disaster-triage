# CLAUDE_BUNDLE.md — upload THIS file to Claude (+ the 5 PNGs listed in §8).
It contains the full paper brief (same as PAPER_HANDOFF.md) plus every data
file pasted verbatim. Nothing else is needed except team names and IEEE formatting.

====================================================================
PART 1 — PAPER BRIEF (instructions for Claude)
====================================================================

You are writing an IEEE-format conference paper for an undergraduate deep-learning
course project (5% of grade, conference track). All facts and numbers below are
measured except where explicitly marked PRELIMINARY/UNMEASURED — never present the
marked ones as results, never invent numbers, never round favorably.

Project in one paragraph: LiveCrisis, a channel-agnostic disaster-triage pipeline.
Free-text reports go to a fine-tuned DistilBERT classifier (category + confidence)
and a strict-JSON LLM planner, rendered on a Leaflet dashboard (map, resources,
NDMA phases, sitrep, contacts). Benchmarked on HumAID crisis tweets with an
event-based split.

Required structure (IEEE): 1. Abstract 2. Introduction 3. Related work (crisis
informatics, BERT/DistilBERT, HumAID/CrisisNLP, LLM structured extraction,
Ushahidi/AIDR/Sahana) 4. Methodology (pipeline, 10-class label set, event split,
training setup) 5. Experiments 6. Results (table + confusion matrices + error
analysis) 7. Discussion (hybrid trade-offs, channel reliability, limitations)
8. Conclusion. Citation rule: only cite references you can verify exist; drop
anything uncertain.

Research questions — RQ1: fine-tuned transformer vs classical baseline vs zero-shot
LLM. RQ2: schema-constrained LLM extraction reliability (PARTIALLY ANSWERED).
RQ3: hybrid design trustworthiness. RQ4 (analytical): offline channel reliability.

BANNED from results (never present as findings): Gemini zero-shot — only 19/200
labels resolved (free-tier quota); mention once as preliminary probe (acc 0.737 /
macro 0.651 on n=19, NOT comparable), never in the comparison table. Pipeline
JSON-validity: 0/50 measured — ALL blocked by quota (HTTP 500s); report as "not
measured," never as 0%. dont_know_cant_judge (11th HumAID label): absent from all
events; task is 10-class. Corroboration score: designed, never field-validated —
future work only. Val set covers 9/10 classes (rarest class train/test-only).

Tone: report losses honestly (BERT margin, beach-day misclassification, quota
failures). Limitations section mandatory: tweet-vs-SMS gap, English-only, small
LLM subset, no field/disconnected test, single seed. Demo ≠ evidence: the
dashboard proves integration; only the benchmark proves the claim.

Per-class reading (for Results/Discussion): all models ace injured_or_dead_people
(~0.93 F1) and struggle with other_relevant_information (~0.45) and
requests_or_urgent_needs (~0.49) — vague overlapping categories. The zero-shot LLM
collapses on rare classes; fine-tuned models hold precision 0.83–0.90 on
missing_or_found_people despite only 358 training examples — the paper's core
argument for fine-tuning (0.698 vs 0.415 macro-F1).

Training setup: DistilBERT — distilbert-base-uncased, 4 epochs, lr 2e-5, max_len
128, fp16, effective batch 32 (16 per-device × 2 grad-accum), class-weighted CE,
best checkpoint by val macro-F1. BERT-base — 3 epochs, 8×4. TF-IDF — 1–2 grams,
20k features, C=1.0, balanced class weights. Seed 42. Hardware: RTX 5060 Laptop
8 GB, driver 617.14, torch 2.11.0+cu128, CUDA 12.8, python 3.12.10.

====================================================================
PART 2 — COMPARISON TABLE (results/comparison_table.md, verbatim)
====================================================================

| model | split | accuracy | macro-F1 | weighted-F1 | latency ms/ex |
|---|---|---|---|---|---|
| tfidf_lr | val | 0.7039 | 0.6748 | 0.7122 | 0.0009021628917839892 |
| tfidf_lr | test | 0.7202 | 0.6352 | 0.7238 | 0.001174533789464596 |
| distilbert | test | 0.7781 | 0.6976 | 0.7751 | 6.801719665527344 |
| bert_base | test | 0.7839 | 0.7064 | 0.7811 | 14.17182207107544 |
| deepseek-v4.1-flash-zero-shot | test-subset-30 | 0.7000 | 0.4152 | 0.7378 |  |

====================================================================
PART 3 — FULL metrics.json (verbatim)
====================================================================

{
 "tfidf_lr_val": {"accuracy": 0.7038594523876274, "macro_f1": 0.6748115698044161,
  "weighted_f1": 0.7121994488837298, "latency_ms_per_example": 0.0009021628917839892},
 "tfidf_lr_test": {"accuracy": 0.7202454940957116, "macro_f1": 0.6351516082197579,
  "weighted_f1": 0.7237892086455571, "latency_ms_per_example": 0.001174533789464596,
  "per_class": {
   "caution_and_advice": {"precision": 0.5106, "recall": 0.4651, "f1": 0.4868},
   "displaced_people_and_evacuations": {"precision": 0.7377, "recall": 0.7082, "f1": 0.7227},
   "infrastructure_and_utility_damage": {"precision": 0.7686, "recall": 0.6478, "f1": 0.7030},
   "injured_or_dead_people": {"precision": 0.8898, "recall": 0.9250, "f1": 0.9071},
   "missing_or_found_people": {"precision": 0.8295, "recall": 0.3687, "f1": 0.5105},
   "not_humanitarian": {"precision": 0.5059, "recall": 0.7825, "f1": 0.6145},
   "other_relevant_information": {"precision": 0.4464, "recall": 0.3709, "f1": 0.4052},
   "requests_or_urgent_needs": {"precision": 0.3312, "recall": 0.6741, "f1": 0.4442},
   "rescue_volunteering_or_donation_effort": {"precision": 0.8970, "recall": 0.7148, "f1": 0.7956},
   "sympathy_and_support": {"precision": 0.7418, "recall": 0.7832, "f1": 0.7619}}},
 "distilbert_test": {"accuracy": 0.7781230577998757, "macro_f1": 0.6976391836675984,
  "weighted_f1": 0.7750783468336432, "latency_ms_per_example_single": 6.801719665527344},
 "distilbert_val_best": {"val_macro_f1": 0.6526199005796582, "batch": [16, 2]},
 "distilbert per-class (P/R/F1)": {
  "caution_and_advice": [0.550, 0.597, 0.572],
  "displaced_people_and_evacuations": [0.771, 0.816, 0.793],
  "infrastructure_and_utility_damage": [0.752, 0.803, 0.776],
  "injured_or_dead_people": [0.901, 0.960, 0.929],
  "missing_or_found_people": [0.833, 0.505, 0.629],
  "not_humanitarian": [0.628, 0.744, 0.681],
  "other_relevant_information": [0.559, 0.377, 0.450],
  "requests_or_urgent_needs": [0.378, 0.671, 0.484],
  "rescue_volunteering_or_donation_effort": [0.902, 0.818, 0.858],
  "sympathy_and_support": [0.768, 0.841, 0.803]},
 "bert_base_test": {"accuracy": 0.7839496581727782, "macro_f1": 0.7064243813736671,
  "weighted_f1": 0.7811462422931199, "latency_ms_per_example_single": 14.17182207107544},
 "bert_base_val_best": {"val_macro_f1": 0.7316256734736964, "batch": [8, 4]},
 "bert_base per-class (P/R/F1)": {
  "caution_and_advice": [0.561, 0.609, 0.584],
  "displaced_people_and_evacuations": [0.734, 0.852, 0.789],
  "infrastructure_and_utility_damage": [0.775, 0.778, 0.776],
  "injured_or_dead_people": [0.901, 0.965, 0.932],
  "missing_or_found_people": [0.904, 0.520, 0.660],
  "not_humanitarian": [0.643, 0.752, 0.693],
  "other_relevant_information": [0.554, 0.393, 0.460],
  "requests_or_urgent_needs": [0.389, 0.668, 0.492],
  "rescue_volunteering_or_donation_effort": [0.899, 0.826, 0.861],
  "sympathy_and_support": [0.800, 0.837, 0.818]},
 "deepseek_zeroshot": {"model": "deepseek-v4.1-flash", "via": "opencode-zen", "n": 30,
  "valid_responses": 30, "invalid_responses": 0, "complete": true, "accuracy": 0.7,
  "macro_f1": 0.4152380952380953, "weighted_f1": 0.7377777777777778},
 "gemini_zeroshot PRELIMINARY": {"model": "gemini-3.6-flash", "n": 200,
  "valid_responses": 19, "failed_quota": 181, "complete": false,
  "valid_accuracy": 0.7368, "valid_macro_f1": 0.6510},
 "gemini_pipeline UNMEASURED": {"n": 50, "http_200": 0, "failed_quota": 50}
}

====================================================================
PART 4 — SPLIT DEFINITION (data/splits/split_summary.json, verbatim)
====================================================================

Labels (10): caution_and_advice, displaced_people_and_evacuations,
infrastructure_and_utility_damage, injured_or_dead_people, missing_or_found_people,
not_humanitarian, other_relevant_information, requests_or_urgent_needs,
rescue_volunteering_or_donation_effort, sympathy_and_support.
Train n=52911 (13 events): greece_wildfires_2018, hurricane_dorian_2019,
hurricane_harvey_2017, hurricane_irma_2017, hurricane_maria_2017,
hurricane_matthew_2016, italy_earthquake_aug_2016, kerala_floods_2018,
maryland_floods_2018, midwestern_us_floods_2019, pakistan_earthquake_2019,
puebla_mexico_earthquake_2017, srilanka_floods_2017.
Val n=10701: hurricane_florence_2018, kaikoura_earthquake_2016,
canada_wildfires_2016 (9/10 classes; missing_or_found_people absent).
Test n=12872 (fully unseen): california_wildfires_2018, cyclone_idai_2019,
ecuador_earthquake_2016.
Train class counts: rescue_volunteering 14981, other_relevant 9360, sympathy 6593,
infrastructure 6434, injured 3967, not_humanitarian 3512, caution 3227, displaced
2470, requests_urgent 2207, missing_or_found 160. Test class counts:
rescue_volunteering 3678, injured 2934, other_relevant 1526, not_humanitarian 1421,
sympathy 1273, infrastructure 846, displaced 425, requests_urgent 313, caution 258,
missing_or_found 198.

====================================================================
PART 5 — TRAINING CURVES (per-epoch val macro-F1; train loss fell throughout)
====================================================================

DistilBERT (batch 16x2): ep1 val_macroF1 0.6435 / ep2 0.6471 / ep3 0.6516 /
ep4 0.6526 (best, kept). Train loss 2.31 -> 0.78.
BERT-base (batch 8x4): ep1 0.6566 / ep2 0.6493 / ep3 0.7316 (best, kept).
Train loss 4.55 -> 1.71.

====================================================================
PART 6 — ERROR ANALYSES (both files, verbatim)
====================================================================

--- results/error_analysis_distilbert.md (2856/12872 wrong) ---
true=other_relevant_information pred=infrastructure_and_utility_damage
(California wildfires): rain/mudslide sympathy post mentioning "fire damaged
areas" — ambiguous, possible label noise.
true=other_relevant_information pred=requests_or_urgent_needs (Idai): recovery
news noting people "still in need of assistance" — genuinely ambiguous.
true=requests_or_urgent_needs pred=rescue_volunteering_or_donation_effort
(Ecuador): "Please donate now! #PrayForEcuador" — donation ask vs urgent need overlap.
true=infrastructure_and_utility_damage pred=injured_or_dead_people (California):
"129 million dead trees" — word "dead" misleads; keyword trap.
true=sympathy_and_support pred=rescue_volunteering_or_donation_effort (Idai):
aerial-assessment charity post — operational content inside sympathy framing.
true=infrastructure_and_utility_damage pred=injured_or_dead_people (Idai):
"more than 200pple died" inside damage discussion — death-count dominates.
true=rescue_volunteering_or_donation_effort pred=sympathy_and_support (Idai):
solidarity concert fundraiser — sympathy framing around a donation event.
true=not_humanitarian pred=other_relevant_information (Idai, x2) and (California):
political blame posts — opinion vs relevance boundary is fuzzy.
true=caution_and_advice pred=other_relevant_information (California): weather
warning with mudslide risk — advice embedded in news.
true=sympathy_and_support pred=injured_or_dead_people (Idai): solidarity message
quoting death toll — numbers dominate.
true=rescue_volunteering_or_donation_effort pred=not_humanitarian (California):
volleyball team wildfire story — human-interest, no explicit ask.
true=other_relevant_information pred=rescue_volunteering_or_donation_effort
(California): "Animal Resources for California Wildfires - RedRover" — short text.
true=rescue_volunteering_or_donation_effort pred=sympathy_and_support (Ecuador):
solidarity campaign announcement — same sympathy/volunteering overlap.
true=other_relevant_information pred=sympathy_and_support (Ecuador): thank-you for
video coverage — gratitude vs sympathy overlap.

--- results/error_analysis_bert_base.md (2781/12872 wrong) ---
Same dominant confusions: sympathy↔rescue_volunteering (fundraiser framing),
not_humanitarian↔other_relevant_information (political opinion boundary),
infrastructure↔injured_or_dead (death counts inside damage reports),
other_relevant→not_humanitarian on utility-criticism posts, caution→other_relevant
on weather/policy news, rescue→infrastructure when damage statistics precede the
donation ask. Short texts (<80 chars) over-represented among errors.

====================================================================
PART 7 — WHAT TO UPLOAD ALONGSIDE THIS FILE
====================================================================

1. The 5 PNGs from results/: confusion_matrix_{tfidf_lr,distilbert,bert_base}.png,
   training_curves_{distilbert,bert_base}.png — ask Claude to reference each in text.
2. Optionally livecrisis-architecture.html (system diagram for the Methodology figure).
3. Nothing else: code, configs, and model files stay in the repo; the paper never
   needed them. Do NOT upload backend/.env (contains API keys).
