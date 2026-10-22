import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
# === ML INTEGRATION (Step 4): local triage classifier ===
# Adds zero hard dependencies: torch/transformers load lazily inside
# src/classify.py, and everything degrades gracefully when models/ is absent.
import sys as _sys
from pathlib import Path as _Path
_ML_SRC = _Path(__file__).resolve().parents[2] / 'src'
if str(_ML_SRC) not in _sys.path:
    _sys.path.insert(0, str(_ML_SRC))


def _local_triage(text):
    try:
        from classify import classify as _clf
        return _clf(text)
    except Exception:
        return None
# === END ML ===
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
# Enable CORS globally for all routes
CORS(app)

# Configure the Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:  #error handling if the API key is not found
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

def clean_gemini_response(text):    #for cleaner response from Gemini API
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
    return text

# === ML: local classifier endpoint (503 when model files absent) ===
@app.route('/classify', methods=['POST'])
def classify_route():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '') or data.get('scenario', '')
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    result = _local_triage(text)
    if result is None:
        return jsonify({'error': 'Local model unavailable'}), 503
    return jsonify(result)
# === END ML ===


@app.route('/analyze', methods=['POST'])
def analyze():
    # Accepts JSON body with fields: scenario (string) and disaster_type (string)
    data = request.get_json(silent=True) or {}
    scenario = data.get("scenario", "")
    disaster_type = data.get("disaster_type", "")

    if not scenario:
        return jsonify({"error": "Scenario is required"}), 400
# prompt is india specific for now, will deal later if need to change to global
    # === ML: triage label as Gemini context (None when model absent) ===
    triage = _local_triage(scenario)
    triage_str = (f"{triage['label']} (confidence {triage['confidence']})"
                  if triage else "unavailable — proceed without it")
    # Setup the prompt to send to Gemini
    prompt = f"""You are a disaster response AI assistant for India. Given the following disaster scenario, return ONLY a valid JSON object with no additional text, no markdown, no code blocks, no backticks. Just raw JSON.

Scenario: {scenario}
Disaster Type: {disaster_type}
Local triage classification: {triage_str}
(Treat the triage label as context, not ground truth; the action plan is still yours.)

Return this exact JSON:
{{
  "incident_location": {{ "lat": float, "lng": float }},
  "severity_title": "e.g. CRITICAL DISASTER ALERT — Flood in Assam",
  "severity_level": "RED or ORANGE or YELLOW",
  "population_affected": integer,
  "estimated_response_time": "e.g. 48-72 hours",
  "resources": [ {{ "type": string, "quantity": integer, "unit": string }} ],
  "ndma_phases": {{
    "immediate": ["action 1", "action 2", "action 3"],
    "short_term": ["action 1", "action 2", "action 3"],
    "recovery": ["action 1", "action 2"]
  }},
  "situation_report": "2 paragraph official sitrep",
  "emergency_contacts": [ {{ "name": string, "number": string, "category": string }} ],
  "map_pins": [ {{ "lat": float, "lng": float, "label": string, "color": "hex like #ff4444" }} ]
}}

Use real Indian district names, real agency names like NDRF and SDMA, realistic numbers. severity_level must be exactly RED, ORANGE, or YELLOW. Return ONLY the JSON."""

    try:
        # Call Gemini API using google-generativeai package with gemini-1.5-flash model
        model = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content(prompt)
        raw_text = response.text
        cleaned = clean_gemini_response(raw_text)
        result = json.loads(cleaned)
        # === ML: echo triage label for the dashboard (may be None) ===
        result['triage'] = triage
        return jsonify(result)

    except Exception as e:
        print(f"Raw Gemini response: {raw_text if 'raw_text' in locals() else 'No response received'}")
        print(f"Error: {e}")
        return jsonify({'error': 'Failed to parse AI response'}), 500

if __name__ == '__main__':
    # Run on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
