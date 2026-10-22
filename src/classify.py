"""Local triage classifier: loads models/distilbert if present, else degrades to None.

classify(text) -> {'label': str, 'confidence': float} | None
"""
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models' / 'distilbert'
MAX_LEN = 128

_model, _tok, _labels, _device = None, None, None, None
_loaded, _attempted = False, False


def _ensure():
    global _model, _tok, _labels, _device, _loaded, _attempted
    if _attempted:
        return _loaded
    _attempted = True
    if not (MODEL_DIR / 'model.safetensors').exists():
        return False
    try:
        import json
        lid = json.loads((MODEL_DIR / 'label_map.json').read_text())
        _labels = sorted(lid, key=lid.get)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR)).to(_device).eval()
        _loaded = True
    except Exception:
        _loaded = False
    return _loaded


def classify(text):
    """Return {'label', 'confidence'} or None when the model is unavailable."""
    if not _ensure():
        return None
    with torch.no_grad():
        inp = _tok(str(text), return_tensors='pt', truncation=True,
                   max_length=MAX_LEN).to(_device)
        probs = torch.softmax(_model(**inp).logits, dim=-1)[0]
        conf, idx = probs.max(0)
        return {'label': _labels[int(idx)], 'confidence': round(float(conf), 4)}


def model_available():
    return _ensure()
