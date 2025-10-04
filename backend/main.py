# main.py
import os
import torch
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import io
import base64
import logging
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vae import ProofVAE
from math_subjects import SUBJECT_ID2NAME
from proof_tactics import TACTIC_NAMES
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------------------------
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Paths & Config
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

# -------------------------------------------------
# Tokenization Helper
def tokenize_text(text: str, vectorizer, seq_len: int, vocab_size: int):
    tokens = vectorizer.build_tokenizer()(text.lower())
    ids = [vectorizer.vocabulary_.get(tok, 0) for tok in tokens]  # 0 = UNK
    ids = (ids + [0] * seq_len)[:seq_len]  # pad/truncate
    return np.array(ids)

# -------------------------------------------------
# Load artifacts
def _load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return joblib.load(path)

try:
    VECTORIZER: CountVectorizer = _load(VECTORIZER_PATH)
    LE_SUBJECT = _load(LE_SUBJECT_PATH)
    LE_TACTIC = _load(LE_TACTIC_PATH)
except FileNotFoundError as e:
    logger.exception("Model artifact load error: %s", e)
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

# -------------------------------------------------
# Load VAE
try:
    VAE = ProofVAE(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN).to(device)
    if os.path.exists(VAE_WEIGHTS):
        VAE.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
        VAE.eval()
        logger.info(f"Loaded VAE weights from {VAE_WEIGHTS}")
    else:
        logger.warning(f"VAE weights not found at {VAE_WEIGHTS}; VAE uninitialized.")
except Exception as e:
    VAE = None
    logger.exception("Failed to initialize VAE: %s", e)

# -------------------------------------------------
# ML Helpers
def id_to_name(cid, names_map, le=None):
    try:
        key = int(cid)
        if names_map and key in names_map:
            return names_map[key]
    except Exception:
        pass
    classes = getattr(le, "classes_", None)
    if classes is not None:
        try:
            return le.inverse_transform(np.array([cid]))[0]
        except Exception:
            pass
    return str(cid)

def _probability(model, X) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[0]
    if hasattr(model, "decision_function"):
        m = model.decision_function(X)
        m = np.atleast_2d(m)
        exp = np.exp(m - m.max())
        return exp / exp.sum()
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

# -------------------------------------------------
# FastAPI App
app = FastAPI(title="Neural Proof Assistant")

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

    X = VECTORIZER.transform([text])
    all_subj = _predict_all(SUBJECT_MODELS, X, LE_SUBJECT, SUBJECT_ID2NAME)
    all_tac = _predict_all(TACTIC_MODELS, X, LE_TACTIC, TACTIC_NAMES)

    subj = os.path.splitext(CHOOSE_SUBJECT)[0]
    tac = os.path.splitext(CHOOSE_TACTIC)[0]

    subj_top3 = _topk(SUBJECT_MODELS.get(subj), X, LE_SUBJECT, SUBJECT_ID2NAME, k=3)
    tac_top3 = _topk(TACTIC_MODELS.get(tac), X, LE_TACTIC, TACTIC_NAMES, k=3)

    latent_vec = None
    if VAE is not None:
        try:
            ids = tokenize_text(text, VECTORIZER, SEQ_LEN, VOCAB_SIZE)
            tokens = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                mu, logvar = VAE.encode(tokens)
                z = VAE.reparameterize(mu, logvar)
            latent_vec = z.cpu().numpy().tolist()
        except Exception:
            logger.exception("VAE embedding failed")

    subj_chart, tac_chart = None, None
    for m in all_subj.values():
        if m.get("distribution"):
            subj_chart = _plot_distribution(m["distribution"], "Subject probabilities")
            break
    for m in all_tac.values():
        if m.get("distribution"):
            tac_chart = _plot_distribution(m["distribution"], "Tactic probabilities")
            break

    return {
        "subject": subj_top3[0]["label"] if subj_top3 else None,
        "tactic": tac_top3[0]["label"] if tac_top3 else None,
        "top3": {"subject": subj_top3, "tactic": tac_top3},
        "all": {"subject": all_subj, "tactic": all_tac},
        "vae_latent": latent_vec,
        "charts": {"subject": subj_chart, "tactic": tac_chart},
    }

@app.post("/vae-embed")
def vae_embed(input: GoalInput):
    text = (input.goal or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input")
    if VAE is None:
        raise HTTPException(status_code=503, detail="VAE not available")
    try:
        ids = tokenize_text(text, VECTORIZER, SEQ_LEN, VOCAB_SIZE)
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            mu, logvar = VAE.encode(tokens)
            z = VAE.reparameterize(mu, logvar)
        return {"latent": z.cpu().numpy().tolist()}
    except Exception:
        logger.exception("VAE embed failed")
        raise HTTPException(status_code=500, detail="VAE embedding failed")

@app.post("/vae-generate")
def vae_generate(input: GoalInput):
    text = (input.goal or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input")
    if VAE is None:
        raise HTTPException(status_code=503, detail="VAE not available")
    try:
        ids = tokenize_text(text, VECTORIZER, SEQ_LEN, VOCAB_SIZE)
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            mu, logvar = VAE.encode(tokens)
            z = VAE.reparameterize(mu, logvar)
            gen_ids = VAE.generate(z, max_length=50, start_token=1, end_token=2)

        inv_vocab = {v: k for k, v in VECTORIZER.vocabulary_.items()}
        gen_tokens = [inv_vocab.get(i, "<UNK>") for i in gen_ids]
        return {"generated": " ".join(gen_tokens)}
    except Exception:
        logger.exception("VAE generation failed")
        raise HTTPException(status_code=500, detail="VAE generation failed")