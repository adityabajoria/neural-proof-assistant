"""
Neural Proof Assistant - FastAPI Backend
Classifies mathematical proof techniques using weak supervision (Snorkel)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import re
from enum import IntEnum
import json

app = FastAPI(
    title="Neural Proof Assistant",
    description="Classify mathematical proof techniques from raw proof text using weak supervision",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PROOF TECHNIQUE TAXONOMY
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


TECHNIQUE_INFO = {
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

# ============================================================================
# SNORKEL LABELING FUNCTIONS
# ============================================================================

class LabelingFunction:
    """Base class for labeling functions with metadata"""
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
    
    def __call__(self, text: str) -> int:
        raise NotImplementedError


class KeywordLF(LabelingFunction):
    """Keyword-based labeling function"""
    def __init__(self, name: str, keywords: List[str], label: int, 
                 negative_keywords: List[str] = None, weight: float = 1.0,
                 require_all: bool = False):
        super().__init__(name, weight)
        self.keywords = [k.lower() for k in keywords]
        self.negative_keywords = [k.lower() for k in (negative_keywords or [])]
        self.label = label
        self.require_all = require_all
    
    def __call__(self, text: str) -> int:
        text_lower = text.lower()
        
        # Check negative keywords first
        for neg in self.negative_keywords:
            if neg in text_lower:
                return ProofTechnique.ABSTAIN
        
        # Check positive keywords
        if self.require_all:
            if all(kw in text_lower for kw in self.keywords):
                return self.label
        else:
            if any(kw in text_lower for kw in self.keywords):
                return self.label
        
        return ProofTechnique.ABSTAIN


class PatternLF(LabelingFunction):
    """Regex pattern-based labeling function"""
    def __init__(self, name: str, patterns: List[str], label: int, 
                 weight: float = 1.0, flags: int = re.IGNORECASE):
        super().__init__(name, weight)
        self.patterns = [re.compile(p, flags) for p in patterns]
        self.label = label
    
    def __call__(self, text: str) -> int:
        for pattern in self.patterns:
            if pattern.search(text):
                return self.label
        return ProofTechnique.ABSTAIN


class StructuralLF(LabelingFunction):
    """Structure-based labeling function (e.g., detecting base case + inductive step)"""
    def __init__(self, name: str, required_sections: List[str], label: int,
                 weight: float = 1.0):
        super().__init__(name, weight)
        self.required_sections = [s.lower() for s in required_sections]
        self.label = label
    
    def __call__(self, text: str) -> int:
        text_lower = text.lower()
        found_count = sum(1 for section in self.required_sections if section in text_lower)
        # Require at least 2/3 of the sections to be present
        if found_count >= len(self.required_sections) * 0.66:
            return self.label
        return ProofTechnique.ABSTAIN


# ============================================================================
# LABELING FUNCTION DEFINITIONS
# ============================================================================

LABELING_FUNCTIONS = [
    # === DIRECT PROOF ===
    KeywordLF(
        "direct_keywords",
        ["we have", "it follows that", "thus we see", "clearly"],
        ProofTechnique.DIRECT,
        negative_keywords=["contradiction", "contradicts", "suppose not", "assume the contrary", "induction", "base case"],
        weight=0.8
    ),
    PatternLF(
        "direct_pattern_assume_show",
        [r"assume\s+.{5,50}\.\s*then", r"let\s+.{3,30}\s+be\s+.{3,30}\.\s*then"],
        ProofTechnique.DIRECT,
        weight=1.0
    ),
    PatternLF(
        "direct_pattern_qed",
        [r"hence\s+.{5,100}(qed|□|∎|proved)", r"therefore\s+.{5,100}(qed|□|∎|proved)"],
        ProofTechnique.DIRECT,
        weight=0.7
    ),
    
    # === CONTRADICTION ===
    KeywordLF(
        "contradiction_explicit",
        ["contradiction", "contradicts", "contradicting"],
        ProofTechnique.CONTRADICTION,
        weight=1.5
    ),
    KeywordLF(
        "contradiction_absurd",
        ["absurd", "impossible", "cannot be"],
        ProofTechnique.CONTRADICTION,
        weight=1.2
    ),
    PatternLF(
        "contradiction_pattern_suppose",
        [
            r"suppose\s+(not|.*\s+not\s+)",
            r"assume\s+(for\s+)?contradiction",
            r"assume\s+the\s+contrary",
            r"suppose\s+.{3,50}\s+were\s+(not\s+)?",
            r"for\s+the\s+sake\s+of\s+contradiction",
            r"suppose.*rational",  # Common pattern for irrationality proofs
            r"suppose.*is\s+rational"
        ],
        ProofTechnique.CONTRADICTION,
        weight=1.3
    ),
    PatternLF(
        "contradiction_pattern_derive",
        [
            r"this\s+(is\s+a\s+)?contradiction",
            r"which\s+contradicts",
            r"contradicting\s+(our|the)\s+(assumption|hypothesis)",
            r"this\s+is\s+absurd",
            r"but\s+this\s+contradicts",
            r"contradicts\s+our\s+assumption",
            r"this\s+contradicts"
        ],
        ProofTechnique.CONTRADICTION,
        weight=1.6
    ),
    PatternLF(
        "contradiction_pattern_therefore",
        [
            r"therefore.*irrational",
            r"therefore.*must\s+be",
            r"thus.*irrational"
        ],
        ProofTechnique.CONTRADICTION,
        weight=1.1
    ),
    
    # === INDUCTION ===
    KeywordLF(
        "induction_explicit",
        ["induction", "inductive"],
        ProofTechnique.INDUCTION,
        weight=1.3
    ),
    StructuralLF(
        "induction_structure",
        ["base case", "inductive step", "inductive hypothesis"],
        ProofTechnique.INDUCTION,
        weight=1.5
    ),
    PatternLF(
        "induction_pattern_base",
        [
            r"base\s+case.*n\s*=\s*[01]",
            r"for\s+n\s*=\s*[01].*holds",
            r"when\s+n\s*=\s*[01]"
        ],
        ProofTechnique.INDUCTION,
        weight=1.2
    ),
    PatternLF(
        "induction_pattern_step",
        [
            r"inductive\s+step",
            r"assume\s+(true\s+)?for\s+n\s*=\s*k",
            r"assume\s+p\s*\(\s*k\s*\)",
            r"suppose\s+(it\s+)?holds?\s+for\s+.{1,20}k",
            r"by\s+(the\s+)?inductive\s+hypothesis"
        ],
        ProofTechnique.INDUCTION,
        weight=1.4
    ),
    PatternLF(
        "induction_pattern_strong",
        [
            r"strong\s+induction",
            r"for\s+all\s+.{1,10}<\s*n",
            r"assume.*holds.*for\s+all.*less\s+than"
        ],
        ProofTechnique.INDUCTION,
        weight=1.3
    ),
    
    # === CONTRAPOSITIVE ===
    KeywordLF(
        "contrapositive_explicit",
        ["contrapositive", "contrapositively"],
        ProofTechnique.CONTRAPOSITIVE,
        weight=1.4
    ),
    PatternLF(
        "contrapositive_pattern",
        [
            r"prove\s+the\s+contrapositive",
            r"by\s+contrapositive",
            r"instead\s+prove\s+.{1,30}not\s+.{1,30}implies\s+.{1,30}not",
            r"we\s+show\s+.{1,20}¬.{1,20}→.{1,20}¬",
            r"equivalently.*if\s+not.*then\s+not"
        ],
        ProofTechnique.CONTRAPOSITIVE,
        weight=1.3
    ),
    
    # === CONSTRUCTION ===
    KeywordLF(
        "construction_explicit",
        ["construct", "constructive", "exhibit"],
        ProofTechnique.CONSTRUCTION,
        negative_keywords=["construct a contradiction"],
        weight=1.1
    ),
    PatternLF(
        "construction_pattern",
        [
            r"we\s+construct",
            r"consider\s+the\s+(following\s+)?(example|construction)",
            r"let\s+.{1,30}\s+be\s+defined\s+(as|by)",
            r"define\s+.{1,30}\s+=",
            r"take\s+.{1,20}=",
            r"such\s+a.{1,20}exists.*namely"
        ],
        ProofTechnique.CONSTRUCTION,
        weight=1.0
    ),
    PatternLF(
        "construction_existence",
        [
            r"to\s+show\s+.{0,20}exists",
            r"we\s+exhibit",
            r"here\s+is\s+(an\s+)?explicit",
            r"the\s+following\s+.{1,20}\s+satisfies"
        ],
        ProofTechnique.CONSTRUCTION,
        weight=1.2
    ),
    
    # === CASES ===
    KeywordLF(
        "cases_explicit",
        ["case 1", "case 2", "case i", "case ii"],
        ProofTechnique.CASES,
        weight=1.3
    ),
    PatternLF(
        "cases_pattern",
        [
            r"we\s+(consider|divide|split)\s+(into\s+)?(two|three|several|the\s+following)\s+cases",
            r"there\s+are\s+(two|three|several)\s+cases",
            r"case\s+\d+\s*:",
            r"case\s+[ivx]+\s*:",
            r"in\s+(the\s+)?(first|second|third)\s+case"
        ],
        ProofTechnique.CASES,
        weight=1.2
    ),
    PatternLF(
        "cases_structure",
        [
            r"either.*or",
            r"if.*is\s+(even|odd|positive|negative)",
            r"suppose.*is\s+(even|odd).*suppose.*is\s+(even|odd)"
        ],
        ProofTechnique.CASES,
        weight=0.9
    ),
    
    # === EXHAUSTION ===
    PatternLF(
        "exhaustion_pattern",
        [
            r"check\s+(each|all|every)",
            r"verify\s+(for\s+)?(each|all|every)",
            r"by\s+exhaustion",
            r"checking\s+all\s+(possible\s+)?cases",
            r"there\s+are\s+only\s+finitely\s+many",
            r"we\s+enumerate"
        ],
        ProofTechnique.EXHAUSTION,
        weight=1.2
    ),
    KeywordLF(
        "exhaustion_keywords",
        ["exhaustively", "finite check", "brute force"],
        ProofTechnique.EXHAUSTION,
        weight=1.1
    ),
]


# ============================================================================
# LABEL MODEL (Simplified Snorkel-style aggregation)
# ============================================================================

class LabelModel:
    """
    Simplified label model that aggregates labeling function outputs
    using weighted voting with accuracy estimation.
    """
    
    def __init__(self, labeling_functions: List[LabelingFunction]):
        self.lfs = labeling_functions
        self.num_classes = len(ProofTechnique) - 1  # Exclude ABSTAIN
        self.lf_accuracies = {lf.name: 0.7 for lf in labeling_functions}  # Prior
    
    def apply(self, texts: List[str]) -> np.ndarray:
        """Apply all labeling functions to texts, return label matrix"""
        L = np.full((len(texts), len(self.lfs)), ProofTechnique.ABSTAIN, dtype=int)
        
        for i, text in enumerate(texts):
            for j, lf in enumerate(self.lfs):
                L[i, j] = lf(text)
        
        return L
    
    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using weighted voting.
        Returns array of shape (n_samples, n_classes)
        """
        n_samples = L.shape[0]
        probs = np.zeros((n_samples, self.num_classes))
        
        for i in range(n_samples):
            votes = np.zeros(self.num_classes)
            total_weight = 0
            
            for j, lf in enumerate(self.lfs):
                label = L[i, j]
                if label != ProofTechnique.ABSTAIN:
                    weight = lf.weight * self.lf_accuracies.get(lf.name, 0.7)
                    votes[label] += weight
                    total_weight += weight
            
            if total_weight > 0:
                probs[i] = votes / total_weight
            else:
                # Uniform prior if no votes
                probs[i] = np.ones(self.num_classes) / self.num_classes
        
        return probs
    
    def predict(self, L: np.ndarray) -> np.ndarray:
        """Return most likely class for each sample"""
        probs = self.predict_proba(L)
        return np.argmax(probs, axis=1)
    
    def get_analysis(self, text: str) -> Dict[str, Any]:
        """Get detailed analysis for a single text"""
        L = self.apply([text])
        probs = self.predict_proba(L)[0]
        
        # Collect triggered labeling functions
        triggered_lfs = []
        for j, lf in enumerate(self.lfs):
            label = L[0, j]
            if label != ProofTechnique.ABSTAIN:
                triggered_lfs.append({
                    "name": lf.name,
                    "label": int(label),
                    "technique": TECHNIQUE_INFO[ProofTechnique(label)]["name"],
                    "weight": lf.weight
                })
        
        # Build probability distribution
        prob_dist = {}
        for i, prob in enumerate(probs):
            technique = ProofTechnique(i)
            info = TECHNIQUE_INFO[technique]
            prob_dist[info["name"]] = {
                "probability": float(prob),
                "color": info["color"],
                "icon": info["icon"],
                "description": info["description"]
            }
        
        predicted_idx = int(np.argmax(probs))
        predicted_technique = ProofTechnique(predicted_idx)
        confidence = float(probs[predicted_idx])
        
        return {
            "predicted_technique": TECHNIQUE_INFO[predicted_technique]["name"],
            "confidence": confidence,
            "technique_info": TECHNIQUE_INFO[predicted_technique],
            "probability_distribution": prob_dist,
            "triggered_labeling_functions": triggered_lfs,
            "coverage": len(triggered_lfs) / len(self.lfs)
        }


# Initialize the label model
label_model = LabelModel(LABELING_FUNCTIONS)


# ============================================================================
# API MODELS
# ============================================================================

class ProofInput(BaseModel):
    text: str
    
class BatchProofInput(BaseModel):
    texts: List[str]

class LabelingFunctionInfo(BaseModel):
    name: str
    type: str
    target_technique: str
    weight: float

class AnalysisResponse(BaseModel):
    predicted_technique: str
    confidence: float
    technique_info: Dict[str, Any]
    probability_distribution: Dict[str, Any]
    triggered_labeling_functions: List[Dict[str, Any]]
    coverage: float


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "Neural Proof Assistant",
        "version": "1.0.0",
        "description": "Classify mathematical proof techniques using weak supervision",
        "endpoints": {
            "/analyze": "POST - Analyze a single proof",
            "/batch": "POST - Analyze multiple proofs",
            "/techniques": "GET - List all proof techniques",
            "/labeling-functions": "GET - List all labeling functions",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.get("/techniques")
async def get_techniques():
    """Return all supported proof techniques"""
    techniques = []
    for technique in ProofTechnique:
        if technique != ProofTechnique.ABSTAIN:
            info = TECHNIQUE_INFO[technique]
            techniques.append({
                "id": int(technique),
                "name": info["name"],
                "description": info["description"],
                "color": info["color"],
                "icon": info["icon"]
            })
    return {"techniques": techniques}


@app.get("/labeling-functions")
async def get_labeling_functions():
    """Return information about all labeling functions"""
    lf_info = []
    for lf in LABELING_FUNCTIONS:
        lf_type = type(lf).__name__
        target = "Unknown"
        
        if hasattr(lf, 'label'):
            target = TECHNIQUE_INFO[ProofTechnique(lf.label)]["name"]
        
        lf_info.append({
            "name": lf.name,
            "type": lf_type,
            "target_technique": target,
            "weight": lf.weight
        })
    
    return {"labeling_functions": lf_info, "total": len(lf_info)}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_proof(proof: ProofInput):
    """Analyze a single proof text and classify its technique"""
    if not proof.text.strip():
        raise HTTPException(status_code=400, detail="Proof text cannot be empty")
    
    analysis = label_model.get_analysis(proof.text)
    return analysis


@app.post("/batch")
async def batch_analyze(batch: BatchProofInput):
    """Analyze multiple proofs at once"""
    if not batch.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    results = []
    for text in batch.texts:
        if text.strip():
            analysis = label_model.get_analysis(text)
            results.append({"text": text[:100] + "..." if len(text) > 100 else text, **analysis})
    
    return {"results": results, "count": len(results)}


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
This is a contradiction, so √2 must be irrational. □"""
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
            "title": "Triangle Inequality",
            "text": """Let a, b, c be the sides of a triangle. We prove |a - b| < c < a + b directly.
Assume we have a valid triangle with sides a, b, c.
The sum of any two sides must be greater than the third side.
Therefore a + b > c, which gives us c < a + b.
Similarly, a + c > b implies c > b - a, and b + c > a implies c > a - b.
Combining these, we get |a - b| < c.
Thus |a - b| < c < a + b, as required. □"""
        },
        {
            "title": "Divisibility by 3",
            "text": """We prove: if 3 does not divide n, then 3 does not divide n².
We prove the contrapositive: if 3 divides n², then 3 divides n.
Suppose 3 | n². Since 3 is prime, by Euclid's lemma, 3 | n or 3 | n.
Either way, 3 | n.
Therefore, by contrapositive, if 3 ∤ n, then 3 ∤ n². □"""
        }
    ]
    return {"demos": demos}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
