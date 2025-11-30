"""
Neural Proof Assistant - FastAPI Backend
Classifies mathematical proof techniques using:
1. Neural model (sentence embeddings + classifier) - Primary
2. Weak supervision (labeling functions) - Fallback

Author: Aditya Bajoria
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import re
from enum import IntEnum

# Import inference module
from inference import (
    predict as neural_predict,
    predict_batch as neural_predict_batch,
    is_model_available,
    get_model_info,
    TECHNIQUE_INFO
)


# ============================================================================
# APP LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    print("\n" + "=" * 50)
    print("🧠 Neural Proof Assistant Starting...")
    print("=" * 50)
    
    if is_model_available():
        print("✅ Neural model found - using ML predictions")
        # Pre-load models
        try:
            from inference import get_encoder, get_classifier, get_label_encoder
            get_encoder()
            get_classifier()
            get_label_encoder()
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            print("   Falling back to rule-based system")
    else:
        print("⚠️ Neural model not found - using rule-based fallback")
        print("   Run the training notebook to enable ML predictions")
    
    print("=" * 50 + "\n")
    yield
    print("\n👋 Shutting down...")


app = FastAPI(
    title="Neural Proof Assistant",
    description="Classify mathematical proof techniques using weak supervision and neural networks",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# FALLBACK: RULE-BASED SYSTEM (Labeling Functions)
# ============================================================================

class ProofTechnique(IntEnum):
    ABSTAIN = -1
    DIRECT = 0
    CONTRADICTION = 1
    INDUCTION = 2
    CONTRAPOSITIVE = 3
    CONSTRUCTION = 4
    CASES = 5
    EXHAUSTION = 6


TECHNIQUE_INFO_LEGACY = {
    ProofTechnique.DIRECT: {
        "name": "Direct Proof",
        "description": "Proves P→Q by assuming P and deriving Q through logical steps",
        "color": "#10B981",
        "icon": "→"
    },
    ProofTechnique.CONTRADICTION: {
        "name": "Proof by Contradiction",
        "description": "Assumes ¬Q and derives a contradiction, proving Q must be true",
        "color": "#EF4444",
        "icon": "⊥"
    },
    ProofTechnique.INDUCTION: {
        "name": "Mathematical Induction",
        "description": "Proves base case P(0), then P(n)→P(n+1) for all n",
        "color": "#8B5CF6",
        "icon": "∀n"
    },
    ProofTechnique.CONTRAPOSITIVE: {
        "name": "Contrapositive",
        "description": "Proves P→Q by proving ¬Q→¬P",
        "color": "#F59E0B",
        "icon": "¬"
    },
    ProofTechnique.CONSTRUCTION: {
        "name": "Proof by Construction",
        "description": "Proves existence by explicitly constructing an example",
        "color": "#06B6D4",
        "icon": "∃"
    },
    ProofTechnique.CASES: {
        "name": "Proof by Cases",
        "description": "Divides the proof into exhaustive cases and proves each",
        "color": "#EC4899",
        "icon": "∨"
    },
    ProofTechnique.EXHAUSTION: {
        "name": "Proof by Exhaustion",
        "description": "Verifies all finite possibilities",
        "color": "#84CC16",
        "icon": "∀"
    }
}


class LabelingFunction:
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
    
    def __call__(self, text: str) -> int:
        raise NotImplementedError


class KeywordLF(LabelingFunction):
    def __init__(self, name: str, keywords: List[str], label: int, 
                 negative_keywords: List[str] = None, weight: float = 1.0):
        super().__init__(name, weight)
        self.keywords = [k.lower() for k in keywords]
        self.negative_keywords = [k.lower() for k in (negative_keywords or [])]
        self.label = label
    
    def __call__(self, text: str) -> int:
        text_lower = text.lower()
        for neg in self.negative_keywords:
            if neg in text_lower:
                return ProofTechnique.ABSTAIN
        if any(kw in text_lower for kw in self.keywords):
            return self.label
        return ProofTechnique.ABSTAIN


class PatternLF(LabelingFunction):
    def __init__(self, name: str, patterns: List[str], label: int, weight: float = 1.0):
        super().__init__(name, weight)
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.label = label
    
    def __call__(self, text: str) -> int:
        for pattern in self.patterns:
            if pattern.search(text):
                return self.label
        return ProofTechnique.ABSTAIN


class StructuralLF(LabelingFunction):
    def __init__(self, name: str, required_sections: List[str], label: int, weight: float = 1.0):
        super().__init__(name, weight)
        self.required_sections = [s.lower() for s in required_sections]
        self.label = label
    
    def __call__(self, text: str) -> int:
        text_lower = text.lower()
        found_count = sum(1 for section in self.required_sections if section in text_lower)
        if found_count >= len(self.required_sections) * 0.66:
            return self.label
        return ProofTechnique.ABSTAIN


# Labeling functions (fallback when neural model unavailable)
LABELING_FUNCTIONS = [
    # DIRECT PROOF
    KeywordLF("direct_keywords", ["we have", "it follows that", "thus we see", "clearly"],
              ProofTechnique.DIRECT,
              negative_keywords=["contradiction", "contradicts", "suppose not", "induction", "base case"],
              weight=0.8),
    PatternLF("direct_pattern", [r"assume\s+.{5,50}\.\s*then", r"let\s+.{3,30}\s+be\s+.{3,30}\.\s*then"],
              ProofTechnique.DIRECT, weight=1.0),
    
    # CONTRADICTION
    KeywordLF("contradiction_explicit", ["contradiction", "contradicts", "contradicting"],
              ProofTechnique.CONTRADICTION, weight=1.5),
    KeywordLF("contradiction_absurd", ["absurd", "impossible", "cannot be"],
              ProofTechnique.CONTRADICTION, weight=1.2),
    PatternLF("contradiction_pattern_suppose",
              [r"suppose\s+(not|.*\s+not\s+)", r"assume\s+(for\s+)?contradiction",
               r"assume\s+the\s+contrary", r"for\s+the\s+sake\s+of\s+contradiction",
               r"suppose.*rational", r"suppose.*is\s+rational"],
              ProofTechnique.CONTRADICTION, weight=1.3),
    PatternLF("contradiction_pattern_derive",
              [r"this\s+(is\s+a\s+)?contradiction", r"which\s+contradicts",
               r"contradicting\s+(our|the)\s+(assumption|hypothesis)", r"this\s+is\s+absurd",
               r"but\s+this\s+contradicts", r"this\s+contradicts"],
              ProofTechnique.CONTRADICTION, weight=1.6),
    
    # INDUCTION
    KeywordLF("induction_explicit", ["induction", "inductive"], ProofTechnique.INDUCTION, weight=1.3),
    StructuralLF("induction_structure", ["base case", "inductive step", "inductive hypothesis"],
                 ProofTechnique.INDUCTION, weight=1.5),
    PatternLF("induction_pattern_base",
              [r"base\s+case.*n\s*=\s*[01]", r"for\s+n\s*=\s*[01].*holds", r"when\s+n\s*=\s*[01]"],
              ProofTechnique.INDUCTION, weight=1.2),
    PatternLF("induction_pattern_step",
              [r"inductive\s+step", r"assume\s+(true\s+)?for\s+n\s*=\s*k",
               r"by\s+(the\s+)?inductive\s+hypothesis"],
              ProofTechnique.INDUCTION, weight=1.4),
    
    # CONTRAPOSITIVE
    KeywordLF("contrapositive_explicit", ["contrapositive", "contrapositively"],
              ProofTechnique.CONTRAPOSITIVE, weight=1.4),
    PatternLF("contrapositive_pattern",
              [r"prove\s+the\s+contrapositive", r"by\s+contrapositive", r"if\s+not.*then\s+not"],
              ProofTechnique.CONTRAPOSITIVE, weight=1.3),
    
    # CONSTRUCTION
    KeywordLF("construction_explicit", ["construct", "constructive", "exhibit"],
              ProofTechnique.CONSTRUCTION, negative_keywords=["construct a contradiction"], weight=1.1),
    PatternLF("construction_pattern",
              [r"we\s+construct", r"we\s+exhibit", r"consider\s+the\s+(following\s+)?example",
               r"such\s+a.{1,20}exists.*namely"],
              ProofTechnique.CONSTRUCTION, weight=1.0),
    
    # CASES
    KeywordLF("cases_explicit", ["case 1", "case 2", "case i", "case ii"],
              ProofTechnique.CASES, weight=1.3),
    PatternLF("cases_pattern",
              [r"we\s+(consider|divide)\s+(into\s+)?(two|three|several)\s+cases",
               r"there\s+are\s+(two|three|several)\s+cases",
               r"case\s+\d+\s*:", r"in\s+(the\s+)?(first|second)\s+case"],
              ProofTechnique.CASES, weight=1.2),
    PatternLF("cases_structure", [r"either.*or", r"suppose.*is\s+(even|odd)"],
              ProofTechnique.CASES, weight=0.9),
    
    # EXHAUSTION
    PatternLF("exhaustion_pattern",
              [r"check\s+(each|all|every)", r"by\s+exhaustion",
               r"checking\s+all\s+(possible\s+)?cases", r"we\s+enumerate"],
              ProofTechnique.EXHAUSTION, weight=1.2),
    KeywordLF("exhaustion_keywords", ["exhaustively", "finite check", "brute force"],
              ProofTechnique.EXHAUSTION, weight=1.1),
]


def rule_based_predict(text: str) -> Dict[str, Any]:
    """Fallback rule-based prediction using labeling functions"""
    votes = {}
    triggered_lfs = []
    
    for lf in LABELING_FUNCTIONS:
        label = lf(text)
        if label != ProofTechnique.ABSTAIN:
            votes[label] = votes.get(label, 0) + lf.weight
            triggered_lfs.append({
                "name": lf.name,
                "label": int(label),
                "technique": TECHNIQUE_INFO_LEGACY[ProofTechnique(label)]["name"],
                "weight": lf.weight
            })
    
    if not votes:
        # Default to Direct Proof if no rules match
        predicted = ProofTechnique.DIRECT
        confidence = 0.3
    else:
        predicted = max(votes.keys(), key=lambda k: votes[k])
        total_weight = sum(votes.values())
        confidence = votes[predicted] / total_weight if total_weight > 0 else 0.5
    
    info = TECHNIQUE_INFO_LEGACY[predicted]
    
    # Build probability distribution
    prob_dist = {}
    for technique in ProofTechnique:
        if technique != ProofTechnique.ABSTAIN:
            t_info = TECHNIQUE_INFO_LEGACY[technique]
            prob = votes.get(technique, 0) / sum(votes.values()) if votes else 0
            prob_dist[t_info["name"]] = {
                "probability": float(prob),
                "color": t_info["color"],
                "icon": t_info["icon"],
                "description": t_info["description"]
            }
    
    return {
        "predicted_technique": info["name"],
        "confidence": confidence,
        "technique_info": info,
        "probability_distribution": prob_dist,
        "triggered_labeling_functions": triggered_lfs,
        "coverage": len(triggered_lfs) / len(LABELING_FUNCTIONS),
        "model_type": "rule-based"
    }


# ============================================================================
# API MODELS
# ============================================================================

class ProofInput(BaseModel):
    text: str
    
class BatchProofInput(BaseModel):
    texts: List[str]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    model_info = get_model_info() if is_model_available() else {"model_available": False}
    return {
        "name": "Neural Proof Assistant",
        "version": "2.0.0",
        "description": "Classify mathematical proof techniques using weak supervision and neural networks",
        "model_status": "neural" if model_info.get("model_available") else "rule-based (fallback)",
        "endpoints": {
            "/analyze": "POST - Analyze a single proof",
            "/batch": "POST - Analyze multiple proofs",
            "/techniques": "GET - List all proof techniques",
            "/model-info": "GET - Model information",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "neural_model_loaded": is_model_available(),
        "fallback_available": True
    }


@app.get("/model-info")
async def model_info():
    """Return information about the loaded model"""
    if is_model_available():
        return get_model_info()
    else:
        return {
            "model_available": False,
            "message": "Neural model not found. Using rule-based fallback.",
            "num_labeling_functions": len(LABELING_FUNCTIONS)
        }


@app.get("/techniques")
async def get_techniques():
    """Return all supported proof techniques"""
    techniques = []
    for technique in ProofTechnique:
        if technique != ProofTechnique.ABSTAIN:
            info = TECHNIQUE_INFO_LEGACY[technique]
            techniques.append({
                "id": int(technique),
                "name": info["name"],
                "description": info["description"],
                "color": info["color"],
                "icon": info["icon"]
            })
    return {"techniques": techniques}


@app.post("/analyze")
async def analyze_proof(proof: ProofInput):
    """Analyze a single proof text and classify its technique"""
    if not proof.text.strip():
        raise HTTPException(status_code=400, detail="Proof text cannot be empty")
    
    # Try neural model first, fall back to rule-based
    if is_model_available():
        try:
            result = neural_predict(proof.text)
            # Add LF analysis for comparison
            rule_result = rule_based_predict(proof.text)
            result["triggered_labeling_functions"] = rule_result["triggered_labeling_functions"]
            result["coverage"] = rule_result["coverage"]
            return result
        except Exception as e:
            print(f"Neural prediction failed: {e}, using fallback")
    
    return rule_based_predict(proof.text)


@app.post("/batch")
async def batch_analyze(batch: BatchProofInput):
    """Analyze multiple proofs at once"""
    if not batch.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    texts = [t for t in batch.texts if t.strip()]
    
    if is_model_available():
        try:
            results = neural_predict_batch(texts)
            return {"results": results, "count": len(results), "model_type": "neural"}
        except Exception as e:
            print(f"Neural batch prediction failed: {e}, using fallback")
    
    # Fallback to rule-based
    results = []
    for text in texts:
        result = rule_based_predict(text)
        result["text"] = text[:100] + "..." if len(text) > 100 else text
        results.append(result)
    
    return {"results": results, "count": len(results), "model_type": "rule-based"}


@app.get("/demo-proofs")
async def get_demo_proofs():
    """Return sample proofs for demonstration"""
    demos = [
        {
            "title": "Irrationality of √2",
            "text": """Suppose, for the sake of contradiction, that √2 is rational. 
Then we can write √2 = a/b where a and b are integers with no common factors. 
Squaring both sides, we get 2 = a²/b², so a² = 2b². 
This means a² is even, so a must be even. 
Write a = 2k for some integer k. Then 4k² = 2b², so b² = 2k². 
But this means b is also even, contradicting our assumption that a and b have no common factors. 
Therefore √2 must be irrational. □"""
        },
        {
            "title": "Sum of First n Natural Numbers",
            "text": """We prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Base case: For n = 1, the left side is 1 and the right side is 1(2)/2 = 1. ✓
Inductive step: Assume the formula holds for n = k, i.e., 1 + 2 + ... + k = k(k+1)/2.
We must show it holds for n = k + 1.
1 + 2 + ... + k + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2.
By the principle of mathematical induction, the formula holds for all positive integers n. □"""
        },
        {
            "title": "Infinitude of Primes",
            "text": """To show there are infinitely many primes, we construct an explicit example.
Given any finite list of primes p₁, p₂, ..., pₙ, consider N = p₁ · p₂ · ... · pₙ + 1.
Either N is prime, or N has a prime factor p. 
But p cannot be any of p₁, ..., pₙ since N leaves remainder 1 when divided by any of them.
Thus we have exhibited a prime not in our original list.
Since this works for any finite list, there must be infinitely many primes. □"""
        },
        {
            "title": "Even or Odd Square",
            "text": """We prove that n² is even if and only if n is even by considering two cases.
Case 1: Suppose n is even. Then n = 2k for some integer k.
Therefore n² = 4k² = 2(2k²), which is even.
Case 2: Suppose n is odd. Then n = 2k + 1 for some integer k.
Therefore n² = 4k² + 4k + 1 = 2(2k² + 2k) + 1, which is odd.
In both cases, n² has the same parity as n. □"""
        },
        {
            "title": "Divisibility by 3",
            "text": """We prove: if 3 does not divide n, then 3 does not divide n².
We prove the contrapositive: if 3 divides n², then 3 divides n.
Suppose 3 | n². Since 3 is prime, by Euclid's lemma, 3 | n or 3 | n.
Either way, 3 | n.
Therefore, by contrapositive, if 3 ∤ n, then 3 ∤ n². □"""
        },
        {
            "title": "Small Perfect Numbers",
            "text": """We verify by exhaustion that 6 is the smallest perfect number.
Check n = 1: divisors sum to 0 ≠ 1.
Check n = 2: divisors sum to 1 ≠ 2.
Check n = 3: divisors sum to 1 ≠ 3.
Check n = 4: divisors sum to 1+2 = 3 ≠ 4.
Check n = 5: divisors sum to 1 ≠ 5.
Check n = 6: divisors sum to 1+2+3 = 6 ✓
Having checked all cases up to 6, we confirm 6 is the smallest perfect number. □"""
        }
    ]
    return {"demos": demos}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
