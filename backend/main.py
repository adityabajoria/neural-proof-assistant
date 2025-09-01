import os
import numpy as np
import joblib
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from math_subjects import SUBJECT_ID2NAME
from proof_tactics import TACTIC_NAMES

## Resolve paths
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.environ.get("NPA_MODELS_DIR", os.path.join(BACKEND_DIR, "models"))

VECTORIZER_PATH = os.path.join(BACKEND_DIR, "vectorizer.pkl")
LE_SUBJECT_PATH = os.path.join(BACKEND_DIR, "label_encoder_subject.pkl")
LE_TACTIC_PATH = os.path.join(BACKEND_DIR, "label_encoder_tactic.pkl")

CHOOSE_SUBJECT = os.environ.get("NPA_SUBJECT_MODEL", "logreg.pkl")
CHOOSE_TACTIC = os.environ.get("NPA_TACTIC_MODEL", "mlp.pkl")

MODELS = ["logreg.pkl", "nb.pkl", "mlp.pkl"]

def _load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

VECTORIZER = _load(VECTORIZER_PATH)
LE_SUBJECT = _load(LE_SUBJECT_PATH)
LE_TACTIC = _load(LE_TACTIC_PATH)

def load_models(head: str):
    folder = os.path.join(MODELS_DIR, head)
    models = {}
    for model in MODELS:
        p = os.path.join(folder, model)
        if os.path.exists(p):
            models[os.path.splitext(model)[0]] = joblib.load(p)
    return models

SUBJECT_MODELS = load_models("subject")
TACTIC_MODELS = load_models("tactic")

# Helper Functons
def _featurize(texts):
    X = VECTORIZER.transform(texts)
    return X


def _decode(le, y, names_map=None):
    # If the encoder was fit on STRING labels, use it directly
    classes = getattr(le, "classes_", None)
    if classes is not None:
        dt = getattr(classes, "dtype", None)
        if hasattr(dt, "kind") and dt.kind in ("U", "S", "O"):
            try:
                return le.inverse_transform(np.array([y]))[0]
            except Exception:
                pass
    # Otherwise the encoder was fit on INTS → use the given id→name map
    if names_map is not None:
        try:
            key = int(y)
            if key in names_map:
                return names_map[key]
        except Exception:
            pass
    # Last resort
    return str(y)

import numpy as np

def id_to_name(cid, names_map, le=None):
    """Always return a readable name for a class id."""
    # Prefer the explicit id→name map (works even if encoder was fit on ints)
    try:
        key = int(cid)
        if names_map and key in names_map:
            return names_map[key]
    except Exception:
        pass

    # If the LabelEncoder was fit on strings, this will return the word
    classes = getattr(le, "classes_", None)
    if classes is not None:
        dt = getattr(classes, "dtype", None)
        if hasattr(dt, "kind") and dt.kind in ("U", "S", "O"):
            try:
                return le.inverse_transform(np.array([cid]))[0]
            except Exception:
                pass

    return str(cid)

def _probability(model, X) -> Optional[float]:
    """Best-effort confidence; fine if it returns None."""
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba is not None and proba.ndim == 2:
                return float(proba.max(axis=1)[0])
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            m = model.decision_function(X)
            m = np.atleast_1d(m)
            if m.ndim == 2:
                m = m.max(axis=1)
            return float(1.0 / (1.0 + np.exp(-m[0])))
        except Exception:
            pass
    return None

def _predict_all(models, X, le, names_map):
    out = {}
    for k, m in models.items():
        cid = m.predict(X)[0]
        out[k] = {
            "id": int(cid) if isinstance(cid, (int, np.integer)) else cid,
            "label": id_to_name(cid, names_map, le),   # <-- map → name
            "proba": _probability(m, X),
        }
    return out

def _topk(model, X, le, names_map, k=3):
    if model is None:
        return []
    proba_fn = getattr(model, "predict_proba", None)
    if callable(proba_fn):
        p = proba_fn(X)
        if p is None or getattr(p, "ndim", 0) != 2 or p.shape[0] == 0:
            return []
        classes = getattr(model, "classes_", None)
        if classes is None:
            return []
        p = p[0]
        idx = np.argsort(p)[::-1][:k]
        return [{
            "id": int(classes[j]) if isinstance(classes[j], (int, np.integer)) else classes[j],
            "label": id_to_name(classes[j], names_map, le),
            "proba": float(p[j]),
        } for j in idx]

    # decision_function fallback (softmaxed margins)
    decision_fn = getattr(model, "decision_function", None)
    if callable(decision_fn):
        s = np.atleast_2d(decision_fn(X))[0]
        classes = getattr(model, "classes_", None)
        if classes is None:
            return []
        idx = np.argsort(s)[::-1][:k]
        exp = np.exp(s - s.max()); sm = exp / exp.sum()
        return [{
            "id": int(classes[j]) if isinstance(classes[j], (int, np.integer)) else classes[j],
            "label": id_to_name(classes[j], names_map, le),
            "proba": float(sm[j]),
        } for j in idx]
    return []


# FASTAPI app
app = FastAPI(title="Neural Proof Assistant - subjects & tactics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GoalInput(BaseModel):
    goal: str

@app.get("/health")
def health():
    return {
        "status": "good",
        "models_dir": MODELS_DIR,
        "subject_models": sorted(list(SUBJECT_MODELS.keys())),
        "tactic_models": sorted(list(TACTIC_MODELS.keys())),
        "chosen_subject": CHOOSE_SUBJECT,
        "chosen_tactic": CHOOSE_TACTIC
    }

@app.post("/predict")
def predict(input: GoalInput):
    text = (input.goal or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input")
    X = _featurize([text])
    all_subj = _predict_all(SUBJECT_MODELS, X, LE_SUBJECT, SUBJECT_ID2NAME) if SUBJECT_MODELS else {}
    all_tac = _predict_all(TACTIC_MODELS, X, LE_TACTIC, TACTIC_NAMES) if TACTIC_MODELS else {}

    if not all_subj and not all_tac:
        raise HTTPException(status_code=503, detail="No models found!")

    subj = os.path.splitext(CHOOSE_SUBJECT)[0]
    tac = os.path.splitext(CHOOSE_TACTIC)[0]

    subj_top3 = _topk(SUBJECT_MODELS.get(subj), X, LE_SUBJECT, SUBJECT_ID2NAME, k=3)
    tac_top3 = _topk(TACTIC_MODELS.get(tac), X, LE_TACTIC, TACTIC_NAMES, k=3)

    subject_label = subj_top3[0]['label'] if subj_top3 else None
    tactic_label = tac_top3[0]['label'] if tac_top3 else None
    subject_id = subj_top3[0]["id"] if subj_top3 else None
    tactic_id = tac_top3[0]["id"] if tac_top3 else None
    return {
        "subject": subject_label,
        "subject_id": subject_id,
        "tactic": tactic_label,
        "tactic_id": tactic_id,
        "top3": {
            "subject": subj_top3,
            "tactic": tac_top3
        },
        "all": {
            "subject": all_subj,
            "tactic": all_tac
        }
    }