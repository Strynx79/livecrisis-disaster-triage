# NOTES.md — Step 0: Existing codebase inspection (no changes made)

## Architecture (as found)

```
index.html (single file, ~822 lines, Bootstrap dark + Leaflet map)
   │  POST /analyze {scenario, disaster_type}
   v
backend/app.py (Flask, port 5000, CORS on)
   │  builds India-specific prompt, calls Gemini, strips markdown, json.loads
   v
Gemini API (code says model "gemini-3.6-flash"; comment still says 1.5-flash — stale comment)
   │  strict-JSON response
   v
Dashboard renders: map pins, resources, NDMA phases, sitrep, contacts
```

- **Backend:** `livecrisis_dashboard/backend/app.py` (85 lines). One route: `POST /analyze`.
  Requires `GEMINI_API_KEY` in `.env` (`load_dotenv()`); warns at startup if missing.
  Deps (`requirements.txt`): `flask, flask-cors, google-generativeai, python-dotenv`.
- **Frontend:** `livecrisis_dashboard/index.html`. "Virtual Crisis Command Center" —
  header bar + sidebar + Leaflet map; fetches `/analyze` and renders fields below.
  (Only head/styles skimmed; full DOM wiring to be mapped during Step 4 integration.)
- **Tests:** `backend/test.py` posts a sample Flood/Assam payload to `/analyze`.
  ⚠️ Pre-existing issue: `test.py` contains an **unresolved git merge-conflict block**
  (`<<<<<<< HEAD ... ======= ... >>>>>>> 9589ca8`) — both sides identical, needs
  marker cleanup before it can run. Left untouched per minimal-edits rule.
- **Docs:** `IDEA.md` (project vision, RQs, architecture, paper outline),
  `backend/README.md` (run instructions), `index.html` sibling `README.md`.

## The JSON schema (enforced via prompt in `app.py`, fields the dashboard expects)

Top-level keys returned by Gemini (raw JSON only, no markdown):
- `incident_location`: `{lat: float, lng: float}`
- `severity_title`: string, e.g. "CRITICAL DISASTER ALERT — Flood in Assam"
- `severity_level`: exactly one of `RED | ORANGE | YELLOW`
- `population_affected`: integer
- `estimated_response_time`: string, e.g. "48-72 hours"
- `resources`: `[{type: string, quantity: int, unit: string}]`
- `ndma_phases`: `{immediate: [3 actions], short_term: [3 actions], recovery: [2 actions]}`
- `situation_report`: 2-paragraph official sitrep
- `emergency_contacts`: `[{name, number, category}]`
- `map_pins`: `[{lat, lng, label, color: hex}]`

There is **no learned model** anywhere in the repo — no training code, no weights,
no `classify` function. The triage "intelligence" is 100% prompt + Gemini.
`clean_gemini_response()` + `json.loads` is the only validation; a parse failure
returns HTTP 500 (`{'error': 'Failed to parse AI response'}`).

## What Step 4 integration must touch (preview, not done yet)

1. New `classify(text) -> {label, confidence}` (new file, e.g. `src/classify.py`).
2. New `POST /classify` route in `app.py` (small, clearly marked edit) + pass the
   label/confidence into the Gemini prompt as context and echo it in the response.
3. `index.html`: display the classifier label/confidence field (small, marked edit).
4. Graceful degradation: everything works as today when `models/best/` is absent.

## GPU / environment (see `results/environment.txt`)

- RTX 5060 Laptop GPU, 8 GB, driver 617.14; torch 2.11.0+cu128 sees `cuda:0`,
  1000×1000 CUDA matmul verified working. Blackwell sm_120 OK with this build.
