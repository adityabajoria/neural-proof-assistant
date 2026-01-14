# Neural Proof Assistant

Project Link: https://neural-proof-assistant.vercel.app

> **Classify mathematical proof techniques from raw proof text using weak supervision + neural networks**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2-blue.svg)](https://reactjs.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

## 🎯 Overview

The Neural Proof Assistant automatically classifies mathematical proofs into their underlying technique categories using a two-stage pipeline:

1. **Weak Supervision** - Labeling functions generate training data programmatically
2. **Neural Model** - Sentence embeddings + classifier learns to generalize

### Supported Proof Techniques

| Technique | Symbol | Description |
|-----------|--------|-------------|
| **Direct Proof** | → | Proves P→Q by assuming P and deriving Q |
| **Contradiction** | ⊥ | Assumes ¬Q and derives a contradiction |
| **Mathematical Induction** | ∀n | Proves base case + inductive step |
| **Contrapositive** | ¬ | Proves P→Q by proving ¬Q→¬P |
| **Construction** | ∃ | Proves existence by explicit construction |
| **Proof by Cases** | ∨ | Divides into exhaustive cases |
| **Exhaustion** | ∀ | Verifies all finite possibilities |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (Google Colab)                      │
├─────────────────────────────────────────────────────────────────┤
│  Labeling Functions → Weak Labels → Sentence Embeddings → MLP  │
│                                          ↓                      │
│                                   classifier.pkl                │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE (Your Computer)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Proof Text → Sentence Transformer → Classifier → Prediction   │
│                  (all-MiniLM-L6-v2)    (from .pkl)              │
│                                                                 │
│  FastAPI Backend ←──────────────────────→ React Frontend        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Without Docker (Recommended for Development)

#### Step 1: Train the Model (Google Colab)

1. Open `notebooks/train.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run all cells (takes ~2 minutes)
3. Download the 3 model files when prompted

#### Step 2: Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your model files
# Copy classifier.pkl, label_encoder.pkl, config.json to backend/models/

# Run server
uvicorn app:app --reload --port 8000
```

#### Step 3: Setup Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

#### Step 4: Open the App

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

### Option 2: With Docker

```bash
# Build and run everything
docker-compose up --build

# Or run in background
docker-compose up -d
```

Then open http://localhost:3000

---

## 📁 Project Structure

```
neural-proof-assistant/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── inference.py        # Model loading & prediction
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile
│   └── models/             # ← Put trained models here
│       ├── classifier.pkl
│       ├── label_encoder.pkl
│       └── config.json
├── frontend/
│   ├── src/
│   │   └── App.jsx         # React application
│   ├── package.json
│   └── Dockerfile
├── notebooks/
│   └── train.ipynb         # Colab training notebook
├── docker-compose.yml
└── README.md
```

---

## 🔬 How It Works

### Weak Supervision Pipeline

Instead of manually labeling thousands of proofs, we use **labeling functions** that encode domain knowledge:

```python
# Example: Detect contradiction proofs
PatternLF(
    "contradiction_pattern",
    patterns=[r"this\s+contradicts", r"which\s+is\s+absurd"],
    label=ProofTechnique.CONTRADICTION
)
```

These functions generate noisy labels that we aggregate using weighted voting.

### Neural Model

1. **Sentence Transformer** (`all-MiniLM-L6-v2`) converts proof text → 384-dim embedding
2. **Classifier** (MLP or Logistic Regression) learns decision boundaries
3. Model generalizes beyond exact pattern matches

### Why This Works

| Approach | Pros | Cons |
|----------|------|------|
| Rules Only | Fast, interpretable | Brittle, exact matches only |
| Neural Only | Generalizes | Needs lots of labeled data |
| **Weak Supervision + Neural** | Best of both | Slightly more complex |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check |
| `GET` | `/techniques` | List proof techniques |
| `GET` | `/model-info` | Model details |
| `POST` | `/analyze` | Analyze single proof |
| `POST` | `/batch` | Analyze multiple proofs |
| `GET` | `/demo-proofs` | Sample proofs |

### Example Request

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Suppose √2 is rational... This contradicts our assumption."}'
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.11 |
| ML | Sentence Transformers, scikit-learn |
| Frontend | React 18, Vite |
| Containerization | Docker, Docker Compose |
| Training | Google Colab (GPU) |

---

## 📈 Future Improvements

- [ ] More training data from ProofWriter/MathLib datasets
- [ ] Multi-label classification (proofs using multiple techniques)
- [ ] Explainability (highlight which parts triggered classification)
- [ ] Integration with Lean/Coq proof assistants

---

## 📄 License

MIT License

---

**Author**: Aditya Bajoria
