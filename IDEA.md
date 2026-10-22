# LiveCrisis: Reliability-Aware Information Extraction for Disaster Response

> Working title: *A Multi-Channel, Reliability-Aware Information Extraction Pipeline for Disaster Response*

## 1. One-paragraph summary

During a disaster, responders receive fragmented, unstructured reports (SMS, radio logs, call transcripts, social media posts) and must turn them into decisions quickly. This project builds a **channel-agnostic pipeline** that converts short free-text reports into structured, auditable output (category, urgency, location, needs) and presents it on a map dashboard. A **fine-tuned transformer classifier** handles triage labeling, an **LLM with a strict JSON schema** generates the action plan, and a **corroboration score** raises confidence when independent reports agree. All models are benchmarked against baselines.

## 2. Problem

- Information arrives late, in fragments, and in inconsistent formats.
- Internet connectivity is often the first thing to fail (power loss, tower damage, congestion), so any solution must accept input from non-internet channels.
- Existing tools usually handle a single channel, need heavy manual curation, or assume connectivity.
- LLM-only pipelines can fabricate locations or counts, and their reliability is rarely measured.

## 3. Research questions

1. **RQ1:** How well does a fine-tuned transformer classify disaster reports compared with classical baselines and a zero-shot LLM?
2. **RQ2:** How reliable is schema-constrained LLM extraction (JSON validity, label agreement with ground truth)?
3. **RQ3:** Does a hybrid design (local classifier + LLM planner + corroboration) give a more trustworthy output than an LLM alone?
4. **RQ4 (analytical):** Which communication channels remain usable when the internet fails, and how reliable is each as an information source?

## 4. Contributions (only claim what is built and measured)

- A benchmark of TF-IDF+LR, (optional BiLSTM), DistilBERT/BERT and zero-shot Gemini on a public crisis dataset, using macro-F1 and an event-based split.
- A measured JSON-validity and label-agreement rate for the existing Gemini pipeline.
- A hybrid architecture that degrades gracefully when the local model is unavailable.
- A literature-based analysis of channel availability and source reliability in disasters.

## 5. System architecture

```
 Channels (SMS, call->speech-to-text, radio log, social post, manual entry)
                          |
                          v
            Normalization: plain short text
                          |
        +-----------------+------------------+
        |                                    |
        v                                    v
 Fine-tuned transformer              Location extraction
 (category + urgency, confidence)    (NER / LLM, then geocoding)
        |                                    |
        +-----------------+------------------+
                          v
          Corroboration score (similar reports
          from independent sources raise confidence)
                          v
        LLM planner (Gemini, strict JSON schema)
        gets label + confidence as context
                          v
      Dashboard: map, resources, action plan, contacts
```

**Design principle:** the LLM is a structured-extraction and planning component, not the sole decision-maker. A measurable, deterministic model handles triage.

## 6. Communication channels and reliability

| Channel | Needs internet | Strengths | Weaknesses |
|---|---|---|---|
| SMS | No | Low network load, store-and-forward | Needs towers, delays, short text |
| Voice calls | No | Rich information | Congests first, hard to structure |
| USSD | No | Works on any phone, structured | Needs operator integration |
| Cell broadcast | No | One-to-many alerts | One-way only |
| Satellite phone / VSAT | No (own link) | Independent of local infrastructure | Cost, few units, power |
| HF/VHF/ham radio | No | Very robust | Low bandwidth, trained operators |
| LoRa / Wi-Fi mesh | No | Phone-to-phone, no towers | Short range, small deployments |
| Social media | Yes | High volume | Noisy, rumors, duplicates |

**Source reliability (qualitative):** official field reports > messages from known or registered contacts > anonymous crowdsourced SMS > social media per item. No single channel is most reliable; reliability comes from **corroboration across sources**.

**Why internet fails:** tower battery backup lasts only hours, physical damage to towers and backhaul, congestion, and in some cases deliberate shutdowns. Failure mode differs by disaster (floods: power and submerged equipment; earthquakes: backhaul; cyclones: towers and power). Verify specifics against sources such as IODA, Cloudflare Radar, OONI and operator reports before citing.

## 7. Resources responders can access

- **Communication:** satellite phones, HAM and VHF radio, mobile command vehicles, Cells-on-Wheels
- **Information:** IMD alerts, CWC flood forecasts, NDMA SACHET, ISRO Bhuvan, Copernicus EMS, SDMA control rooms, helpline 112
- **Human:** NDRF/SDRF, local administration, volunteers, NGOs
- **Digital:** drones, GIS tools, social media monitoring

(Verify against NDMA and NDRF documents before citing.)

## 8. AI and deep learning techniques

**Implemented in this project**
- Fine-tuned transformer text classification (DistilBERT, optionally BERT-base)
- Classical baseline (TF-IDF + Logistic Regression), optional BiLSTM
- Zero-shot LLM extraction with JSON schema constraint
- Location extraction via NER and geocoding (if time allows)

**Future scope only (do not claim as built)**
- Speech recognition for voice channels (Whisper)
- Multilingual and code-mixed text (MuRIL, IndicBERT, XLM-R)
- Satellite damage detection with CNN/U-Net (e.g. xBD dataset)
- Credibility and misinformation detection
- Multimodal image + text (CrisisMMD)
- Flood forecasting with LSTMs

## 9. Tech stack

- **Ingestion:** SMS gateway (Twilio or GSM modem), speech-to-text, manual text input
- **Backend:** Python, FastAPI (or existing framework in the repo)
- **ML:** PyTorch, HuggingFace Transformers, scikit-learn
- **LLM:** Gemini with JSON schema output
- **Geo:** Nominatim or GeoNames, Leaflet
- **Frontend:** existing HTML dashboard

## 10. Datasets and evaluation

- **Datasets (verify names, licenses and URLs):** HumAID, CrisisNLP, CrisisMMD; fallback Kaggle "NLP with Disaster Tweets"
- **Split:** by disaster event when metadata exists, to test generalization to unseen events; fixed seed 42
- **Metrics:** accuracy, macro-F1 (primary), weighted-F1, per-class precision/recall/F1, confusion matrix, mean inference latency (state the hardware)
- **Gemini-specific:** JSON-valid rate, label agreement with ground truth, on a fixed test subset
- **Rule:** report results honestly, including cases where the fine-tuned model loses to a baseline or Gemini

## 11. Existing solutions

| Solution | Benefit | Drawback |
|---|---|---|
| Ushahidi | Crowdsourced mapping, widely deployed | Manual curation, little ML triage |
| AIDR (QCRI) | Tweet classification for crises | Social-media focused, needs human labels |
| Google SOS Alerts / Flood Forecasting | Scale and data | Needs internet, not a responder-side triage tool |
| NDMA SACHET | Official alerts in India | Mostly one-way broadcast |
| Sahana Eden | Full disaster management suite | Heavy to deploy |
| Copernicus EMS / UNOSAT | Authoritative satellite maps | Delay of hours to days, expert-driven |
| Meshtastic / goTenna | Works offline | Low bandwidth, no analytics |

**Gap:** most tools handle one channel, need heavy training, or depend on the internet. This project offers a lightweight, channel-agnostic extraction layer with structured, auditable output.

## 12. Things to avoid (and why)

- **Unmeasured claims** ("3 seconds", "no hallucinations"): measure them or call them estimates.
- **LLM alone for life-critical output:** it can fabricate locations or counts; validate with NER and geocoding.
- **Single-channel dependence:** it fails when that channel fails.
- **Synthetic data as evidence:** label demo data clearly as a demo.
- **Sending victim data to third-party APIs:** note the privacy risk; propose local models as mitigation.
- **Accuracy as the only metric:** disaster data is imbalanced, so use macro-F1.
- **Random splits that leak events:** split by event.
- **Scope creep:** satellite, mesh and multi-agency sync stay in Future Scope.
- **Claiming offline capability that was not built.**

## 13. Limitations (to state in the paper)

- Trained on public tweet data, which differs from real SMS and radio reports.
- English-centric; Hindi and code-mixed text not covered.
- Small Gemini evaluation subset; results depend on API model version.
- Corroboration score is a heuristic, not validated in the field.
- Not tested in a real disaster or on a disconnected network.

## 14. Future scope

1. Live ingestion (SMS, USSD, radio logs, social feeds)
2. Satellite damage assessment (Sentinel-2, Bhuvan)
3. Multi-agency real-time sync with role-based access
4. Offline mesh input (LoRa or WebRTC)
5. One-click SITREP PDF export
6. Fine-tuning on India-specific disaster data and multilingual support

## 15. Paper outline (IEEE format)

1. Abstract
2. Introduction (problem and motivation)
3. Related work (crisis informatics, BERT/DistilBERT, HumAID/CrisisNLP, LLM structured extraction, existing platforms)
4. Methodology (pipeline, models, JSON schema, corroboration score)
5. Experiments (datasets, split, training setup, metrics)
6. Results (comparison table, confusion matrix, error analysis)
7. Discussion (channel reliability, hybrid design trade-offs, limitations)
8. Conclusion and future scope

**Citation rule:** verify every reference exists and is cited correctly before including it. Do not rely on reference lists from memory.

## 16. Deliverables

- Working dashboard with `/classify` integration
- Trained model, comparison table, confusion matrices, training curves, error analysis
- Reproducible scripts (`run_all.sh`, `requirements.txt`, config)
- IEEE-format paper
- Demo line: *"A working proof of concept; the pipeline stays the same, only data sources and connected agencies change."*
