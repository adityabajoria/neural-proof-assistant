# app.py
import os
import torch
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # headless-friendly backend
import matplotlib.pyplot as plt
import io
import base64
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vae import ProofVAE
from math_subjects import SUBJECT_ID2NAME
from proof_tactics import TACTIC_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## Resolve paths
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.environ.get("NPA_MODELS_DIR", os.path.join(BACKEND_DIR, "models"))

VECTORIZER_PATH = os.path.join(BACKEND_DIR, "vectorizer.pkl")
LE_SUBJECT_PATH = os.path.join(BACKEND_DIR, "label_encoder_subject.pkl")
LE_TACTIC_PATH = os.path.join(BACKEND_DIR, "label_encoder_tactic.pkl")

CHOOSE_SUBJECT = os.environ.get("NPA_SUBJECT_MODEL", "logreg.pkl")
CHOOSE_TACTIC = os.environ.get("NPA_TACTIC_MODEL", "mlp.pkl")

MODELS = ["logreg.pkl", "nb.pkl", "mlp.pkl"]

# VAE configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = int(os.environ.get("VAE_VOCAB_SIZE", 1000))
SEQ_LEN = int(os.environ.get("VAE_SEQ_LEN", 50))
VAE_WEIGHTS = os.environ.get("VAE_WEIGHTS", os.path.join(BACKEND_DIR, "models", "proofvae.pt"))

# Create VAE instance (but don't crash if weights missing)
try:
    VAE = ProofVAE(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN).to(device)
    if os.path.exists(VAE_WEIGHTS):
        VAE.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
        VAE.eval()
        logger.info(f"Loaded VAE weights from {VAE_WEIGHTS}")
    else:
        logger.warning(f"VAE weights not found at {VAE_WEIGHTS}; VAE will be uninitialized for inference.")
except Exception as e:
    VAE = None
    logger.exception("Failed to initialize VAE: %s", e)

def _load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return joblib.load(path)

# Load vectorizer / label encoders with clear errors
try:
    VECTORIZER = _load(VECTORIZER_PATH)
    LE_SUBJECT = _load(LE_SUBJECT_PATH)
    LE_TACTIC = _load(LE_TACTIC_PATH)
except FileNotFoundError as e:
    logger.exception("Model artifact load error: %s", e)
    # re-raise so app startup fails loudly and you can fix artifacts
    raise

def load_models(head: str):
    folder = os.path.join(MODELS_DIR, head)
    models = {}
    for model in MODELS:
        p = os.path.join(folder, model)
        if os.path.exists(p):
            models[os.path.splitext(model)[0]] = joblib.load(p)
            logger.info("Loaded model: %s", p)
    return models

SUBJECT_MODELS = load_models("subject")
TACTIC_MODELS = load_models("tactic")

# ---------------- Helper Functions ----------------
def _featurize(texts):
    return VECTORIZER.transform(texts)

def id_to_name(cid, names_map, le=None):
    try:
        key = int(cid)
        if names_map and key in names_map:
            return names_map[key]
    except Exception:
        pass
    classes = getattr(le, "classes_", None)
    if classes is not None:
        dt = getattr(classes, "dtype", None)
        if hasattr(dt, "kind") and dt.kind in ("U", "S", "O"):
            try:
                return le.inverse_transform(np.array([cid]))[0]
            except Exception:
                pass
    return str(cid)

def _probability(model, X) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[0]  # return full distribution
        except Exception:
            logger.debug("predict_proba failed for model %s", model, exc_info=True)
            pass
    if hasattr(model, "decision_function"):
        try:
            m = model.decision_function(X)
            m = np.atleast_2d(m)
            exp = np.exp(m - m.max())
            return exp / exp.sum()
        except Exception:
            logger.debug("decision_function fallback failed for %s", model, exc_info=True)
            pass
    return None

def _predict_all(models, X, le, names_map):
    out = {}
    for k, m in models.items():
        try:
            cid = m.predict(X)[0]
            probas = _probability(m, X)
            distribution = {}
            if probas is not None:
                classes = getattr(m, "classes_", [])
                for i, cls in enumerate(classes):
                    distribution[id_to_name(cls, names_map, le)] = float(probas[i])
            out[k] = {
                "id": int(cid) if isinstance(cid, (int, np.integer)) else cid,
                "label": id_to_name(cid, names_map, le),
                "proba": float(probas.max()) if probas is not None else None,
                "distribution": distribution
            }
        except Exception:
            logger.exception("Error predicting with model %s", k)
            out[k] = {"error": "prediction failed"}
    return out

def _topk(model, X, le, names_map, k=3):
    try:
        probas = _probability(model, X)
        classes = getattr(model, "classes_", None)
        if probas is None or classes is None:
            return []
        idx = np.argsort(probas)[::-1][:k]
        return [{
            "id": int(classes[j]) if isinstance(classes[j], (int, np.integer)) else classes[j],
            "label": id_to_name(classes[j], names_map, le),
            "proba": float(probas[j]),
        } for j in idx]
    except Exception:
        logger.exception("topk failure")
        return []

def _plot_distribution(distribution: Dict[str, float], title: str) -> Optional[str]:
    if not distribution:
        return None
    labels = list(distribution.keys())
    values = list(distribution.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ---------------- FASTAPI ----------------
app = FastAPI(title="Neural Proof Assistant - subjects & tactics + VAE")

# CORS: add common dev origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GoalInput(BaseModel):
    goal: str

@app.get("/")
def root():
    return {"message": "Neural Proof Assistant Backend is running 🚀"}

@app.get("/health")
def health():
    return {
        "status": "good",
        "models_dir": MODELS_DIR,
        "subject_models": sorted(list(SUBJECT_MODELS.keys())),
        "tactic_models": sorted(list(TACTIC_MODELS.keys())),
        "chosen_subject": CHOOSE_SUBJECT,
        "chosen_tactic": CHOOSE_TACTIC,
        "vae_loaded": (VAE is not None and os.path.exists(VAE_WEIGHTS)),
    }

@app.post("/predict")
def predict(input: GoalInput):
    text = (input.goal or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input")

    # Featurize for sklearn models
    try:
        X = _featurize([text])
    except Exception:
        logger.exception("Featurization failed")
        raise HTTPException(status_code=500, detail="Featurization failed")

    all_subj = _predict_all(SUBJECT_MODELS, X, LE_SUBJECT, SUBJECT_ID2NAME) if SUBJECT_MODELS else {}
    all_tac = _predict_all(TACTIC_MODELS, X, LE_TACTIC, TACTIC_NAMES) if TACTIC_MODELS else {}

    if not all_subj and not all_tac:
        raise HTTPException(status_code=503, detail="No models found!")

    subj = os.path.splitext(CHOOSE_SUBJECT)[0]
    tac = os.path.splitext(CHOOSE_TACTIC)[0]

    subj_top3 = _topk(SUBJECT_MODELS.get(subj), X, LE_SUBJECT, SUBJECT_ID2NAME, k=3)
    tac_top3 = _topk(TACTIC_MODELS.get(tac), X, LE_TACTIC, TACTIC_NAMES, k=3)

    # VAE embedding — use encode() + reparameterize() to obtain z
    latent_vec = None
    if VAE is not None:
        try:
            # TODO: replace tokenizer: map `text` -> token ids of length <= SEQ_LEN
            # For now we pad/truncate a naive tokenization (placeholder)
            # WARNING: this is a placeholder — replace with your tokenizer
            toks = np.random.randint(0, VOCAB_SIZE, size=(1, SEQ_LEN))
            tokens = torch.tensor(toks, dtype=torch.long, device=device)
            with torch.no_grad():
                mu, logvar = VAE.encode(tokens)         # mu/logvar shape: (1, latent_dim)
                z = VAE.reparameterize(mu, logvar)     # (1, latent_dim)
            latent_vec = z.cpu().numpy().tolist()
        except Exception:
            logger.exception("VAE embedding failed")
            latent_vec = None

    # Probability visualization charts (base64 PNGs)
    subj_chart = None
    tac_chart = None

    # The structures in all_subj/all_tac map model-name->{"distribution":{label:prob,...}}
    # Choose first available distribution for subject and tactic to plot.
    try:
        # find a distribution to visualize for subject
        for m in all_subj.values():
            dist = m.get("distribution", {})
            if dist:
                subj_chart = _plot_distribution(dist, "Subject probabilities")
                break
        # find a distribution to visualize for tactic
        for m in all_tac.values():
            dist = m.get("distribution", {})
            if dist:
                tac_chart = _plot_distribution(dist, "Tactic probabilities")
                break
    except Exception:
        logger.exception("Chart generation failed")

    return {
        "subject": subj_top3[0]["label"] if subj_top3 else None,
        "subject_id": subj_top3[0]["id"] if subj_top3 else None,
        "tactic": tac_top3[0]["label"] if tac_top3 else None,
        "tactic_id": tac_top3[0]["id"] if tac_top3 else None,
        "top3": {"subject": subj_top3, "tactic": tac_top3},
        "all": {"subject": all_subj, "tactic": all_tac},
        "vae_latent": latent_vec,
        "charts": {"subject": subj_chart, "tactic": tac_chart},
    }

@app.post("/vae-embed")
def vae_embed(input: GoalInput):
    """Return a VAE latent embedding for the input text (inference-only).
       NOTE: this uses a placeholder tokenization — replace with your tokenizer."""
    text = (input.goal or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input")
    if VAE is None:
        raise HTTPException(status_code=503, detail="VAE not available")

    try:
        toks = np.random.randint(0, VOCAB_SIZE, size=(1, SEQ_LEN))
        tokens = torch.tensor(toks, dtype=torch.long, device=device)
        with torch.no_grad():
            mu, logvar = VAE.encode(tokens)
            z = VAE.reparameterize(mu, logvar)
        return {"latent": z.cpu().numpy().tolist()}
    except Exception:
        logger.exception("VAE embed failed")
        raise HTTPException(status_code=500, detail="VAE embedding failed")