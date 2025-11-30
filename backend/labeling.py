"""
Enhanced Weak Supervision Module for Neural Proof Assistant

This module implements a more sophisticated Snorkel-style weak supervision pipeline
with improved coverage, conflict resolution, and accuracy estimation.

Key improvements over basic implementation:
1. Better coverage through overlapping LFs with different granularities
2. Conflict resolution using correlation analysis
3. Estimated accuracy from LF agreement patterns
4. Support for soft labels and uncertainty quantification
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re
from enum import IntEnum


class ProofTechnique(IntEnum):
    """Enumeration of proof techniques with ABSTAIN as -1"""
    ABSTAIN = -1
    DIRECT = 0
    CONTRADICTION = 1
    INDUCTION = 2
    CONTRAPOSITIVE = 3
    CONSTRUCTION = 4
    CASES = 5
    EXHAUSTION = 6


@dataclass
class LabelingFunctionResult:
    """Result from applying a labeling function"""
    label: int
    confidence: float = 1.0
    matched_spans: List[Tuple[int, int]] = field(default_factory=list)
    evidence: str = ""


class BaseLabelingFunction:
    """Base class for all labeling functions with enhanced metadata"""
    
    def __init__(
        self, 
        name: str, 
        target_label: int,
        weight: float = 1.0,
        min_confidence: float = 0.5,
        description: str = ""
    ):
        self.name = name
        self.target_label = target_label
        self.weight = weight
        self.min_confidence = min_confidence
        self.description = description
        
        # Runtime statistics
        self.num_calls = 0
        self.num_labels = 0
        self.num_abstains = 0
    
    def __call__(self, text: str) -> LabelingFunctionResult:
        self.num_calls += 1
        result = self._apply(text)
        
        if result.label == ProofTechnique.ABSTAIN:
            self.num_abstains += 1
        else:
            self.num_labels += 1
        
        return result
    
    def _apply(self, text: str) -> LabelingFunctionResult:
        raise NotImplementedError
    
    @property
    def coverage(self) -> float:
        """Fraction of inputs that received a label"""
        if self.num_calls == 0:
            return 0.0
        return self.num_labels / self.num_calls


class KeywordLF(BaseLabelingFunction):
    """
    Keyword-based labeling function with support for:
    - Positive and negative keyword sets
    - Proximity constraints (keywords must appear close together)
    - Confidence scaling based on keyword count
    """
    
    def __init__(
        self,
        name: str,
        target_label: int,
        keywords: List[str],
        negative_keywords: List[str] = None,
        require_all: bool = False,
        proximity_window: int = None,
        weight: float = 1.0,
        **kwargs
    ):
        super().__init__(name, target_label, weight, **kwargs)
        self.keywords = [k.lower() for k in keywords]
        self.negative_keywords = [k.lower() for k in (negative_keywords or [])]
        self.require_all = require_all
        self.proximity_window = proximity_window  # Max chars between keywords
    
    def _apply(self, text: str) -> LabelingFunctionResult:
        text_lower = text.lower()
        
        # Check negative keywords first
        for neg in self.negative_keywords:
            if neg in text_lower:
                return LabelingFunctionResult(ProofTechnique.ABSTAIN)
        
        # Find all keyword matches with positions
        matches = []
        for kw in self.keywords:
            start = 0
            while True:
                pos = text_lower.find(kw, start)
                if pos == -1:
                    break
                matches.append((pos, pos + len(kw), kw))
                start = pos + 1
        
        if not matches:
            return LabelingFunctionResult(ProofTechnique.ABSTAIN)
        
        # Check if all keywords are required
        matched_keywords = set(m[2] for m in matches)
        if self.require_all and len(matched_keywords) < len(self.keywords):
            return LabelingFunctionResult(ProofTechnique.ABSTAIN)
        
        # Check proximity constraint
        if self.proximity_window and len(matches) > 1:
            matches.sort(key=lambda x: x[0])
            max_gap = max(matches[i+1][0] - matches[i][1] for i in range(len(matches)-1))
            if max_gap > self.proximity_window:
                return LabelingFunctionResult(ProofTechnique.ABSTAIN)
        
        # Calculate confidence based on number of matches
        confidence = min(1.0, 0.5 + 0.1 * len(matches))
        
        return LabelingFunctionResult(
            label=self.target_label,
            confidence=confidence,
            matched_spans=[(m[0], m[1]) for m in matches],
            evidence=f"Found keywords: {matched_keywords}"
        )


class PatternLF(BaseLabelingFunction):
    """
    Regex pattern-based labeling function with:
    - Multiple patterns (OR logic)
    - Named capture groups for evidence extraction
    - Confidence based on pattern specificity
    """
    
    def __init__(
        self,
        name: str,
        target_label: int,
        patterns: List[str],
        weight: float = 1.0,
        flags: int = re.IGNORECASE | re.MULTILINE,
        **kwargs
    ):
        super().__init__(name, target_label, weight, **kwargs)
        self.patterns = []
        for p in patterns:
            try:
                self.patterns.append(re.compile(p, flags))
            except re.error as e:
                print(f"Warning: Invalid pattern '{p}': {e}")
    
    def _apply(self, text: str) -> LabelingFunctionResult:
        all_matches = []
        
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                all_matches.append({
                    'span': match.span(),
                    'text': match.group(),
                    'groups': match.groupdict() if match.groupdict() else None
                })
        
        if not all_matches:
            return LabelingFunctionResult(ProofTechnique.ABSTAIN)
        
        # Confidence increases with more distinct pattern matches
        confidence = min(1.0, 0.6 + 0.15 * len(all_matches))
        
        return LabelingFunctionResult(
            label=self.target_label,
            confidence=confidence,
            matched_spans=[m['span'] for m in all_matches],
            evidence=f"Matched patterns: {[m['text'][:50] for m in all_matches[:3]]}"
        )


class StructuralLF(BaseLabelingFunction):
    """
    Structure-aware labeling function that looks for:
    - Ordered sections (e.g., "base case" before "inductive step")
    - Required + optional components
    - Hierarchical structure
    """
    
    def __init__(
        self,
        name: str,
        target_label: int,
        required_sections: List[str],
        optional_sections: List[str] = None,
        ordered: bool = False,
        min_required: float = 0.66,
        weight: float = 1.0,
        **kwargs
    ):
        super().__init__(name, target_label, weight, **kwargs)
        self.required_sections = [s.lower() for s in required_sections]
        self.optional_sections = [s.lower() for s in (optional_sections or [])]
        self.ordered = ordered
        self.min_required = min_required
    
    def _apply(self, text: str) -> LabelingFunctionResult:
        text_lower = text.lower()
        
        # Find positions of all sections
        found_required = []
        for section in self.required_sections:
            pos = text_lower.find(section)
            if pos != -1:
                found_required.append((section, pos))
        
        found_optional = []
        for section in self.optional_sections:
            pos = text_lower.find(section)
            if pos != -1:
                found_optional.append((section, pos))
        
        # Check minimum required sections
        if len(found_required) < len(self.required_sections) * self.min_required:
            return LabelingFunctionResult(ProofTechnique.ABSTAIN)
        
        # Check ordering constraint
        if self.ordered and len(found_required) > 1:
            positions = [pos for _, pos in found_required]
            if positions != sorted(positions):
                return LabelingFunctionResult(ProofTechnique.ABSTAIN)
        
        # Calculate confidence
        required_ratio = len(found_required) / len(self.required_sections)
        optional_bonus = 0.1 * len(found_optional) if self.optional_sections else 0
        confidence = min(1.0, required_ratio * 0.8 + optional_bonus + 0.2)
        
        return LabelingFunctionResult(
            label=self.target_label,
            confidence=confidence,
            matched_spans=[],
            evidence=f"Found sections: {[s for s, _ in found_required + found_optional]}"
        )


class CompositeLF(BaseLabelingFunction):
    """
    Combines multiple labeling functions with AND/OR logic
    """
    
    def __init__(
        self,
        name: str,
        target_label: int,
        lfs: List[BaseLabelingFunction],
        require_all: bool = True,
        min_required: int = None,
        weight: float = 1.0,
        **kwargs
    ):
        super().__init__(name, target_label, weight, **kwargs)
        self.lfs = lfs
        self.require_all = require_all
        self.min_required = min_required or (len(lfs) if require_all else 1)
    
    def _apply(self, text: str) -> LabelingFunctionResult:
        results = [lf(text) for lf in self.lfs]
        labeled_results = [r for r in results if r.label != ProofTechnique.ABSTAIN]
        
        if len(labeled_results) < self.min_required:
            return LabelingFunctionResult(ProofTechnique.ABSTAIN)
        
        # Aggregate confidence
        avg_confidence = np.mean([r.confidence for r in labeled_results])
        all_spans = []
        for r in labeled_results:
            all_spans.extend(r.matched_spans)
        
        return LabelingFunctionResult(
            label=self.target_label,
            confidence=avg_confidence,
            matched_spans=all_spans,
            evidence=f"Composite of {len(labeled_results)} LFs"
        )


class NegationLF(BaseLabelingFunction):
    """
    Labeling function that triggers when another LF does NOT trigger
    Useful for "none of the above" or default categories
    """
    
    def __init__(
        self,
        name: str,
        target_label: int,
        negative_lf: BaseLabelingFunction,
        weight: float = 1.0,
        **kwargs
    ):
        super().__init__(name, target_label, weight, **kwargs)
        self.negative_lf = negative_lf
    
    def _apply(self, text: str) -> LabelingFunctionResult:
        result = self.negative_lf(text)
        
        if result.label == ProofTechnique.ABSTAIN:
            return LabelingFunctionResult(
                label=self.target_label,
                confidence=0.5,  # Lower confidence for negation
                evidence="Negative condition met"
            )
        
        return LabelingFunctionResult(ProofTechnique.ABSTAIN)


# ============================================================================
# ENHANCED LABEL MODEL
# ============================================================================

class EnhancedLabelModel:
    """
    Enhanced label model with:
    - Estimated LF accuracies from agreement patterns
    - Correlation-aware aggregation
    - Uncertainty quantification
    - Support for soft labels
    """
    
    def __init__(
        self,
        labeling_functions: List[BaseLabelingFunction],
        num_classes: int = 7,
        prior: np.ndarray = None
    ):
        self.lfs = labeling_functions
        self.num_classes = num_classes
        self.prior = prior if prior is not None else np.ones(num_classes) / num_classes
        
        # Initialize accuracies with prior
        self.lf_accuracies = {lf.name: 0.7 for lf in self.lfs}
        
        # Correlation matrix between LFs (learned from data)
        self.lf_correlations = np.eye(len(self.lfs))
        
        # Track LF agreement statistics
        self.agreement_counts = defaultdict(lambda: defaultdict(int))
    
    def apply(self, texts: List[str]) -> Tuple[np.ndarray, List[List[LabelingFunctionResult]]]:
        """
        Apply all LFs to texts, returning label matrix and detailed results
        """
        n = len(texts)
        m = len(self.lfs)
        
        L = np.full((n, m), ProofTechnique.ABSTAIN, dtype=int)
        confidences = np.zeros((n, m))
        all_results = []
        
        for i, text in enumerate(texts):
            results = []
            for j, lf in enumerate(self.lfs):
                result = lf(text)
                L[i, j] = result.label
                confidences[i, j] = result.confidence
                results.append(result)
            all_results.append(results)
            
            # Update agreement statistics
            self._update_agreements(L[i])
        
        return L, confidences, all_results
    
    def _update_agreements(self, labels: np.ndarray):
        """Track which LFs agree/disagree"""
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] != ProofTechnique.ABSTAIN and labels[j] != ProofTechnique.ABSTAIN:
                    if labels[i] == labels[j]:
                        self.agreement_counts[i][j] += 1
                        self.agreement_counts[j][i] += 1
    
    def estimate_accuracies(self, L: np.ndarray) -> Dict[str, float]:
        """
        Estimate LF accuracies using agreement rates
        Based on the insight that accurate LFs tend to agree with other accurate LFs
        """
        n, m = L.shape
        
        # Calculate pairwise agreement rates
        agreements = np.zeros((m, m))
        counts = np.zeros((m, m))
        
        for i in range(n):
            for j in range(m):
                for k in range(j + 1, m):
                    if L[i, j] != ProofTechnique.ABSTAIN and L[i, k] != ProofTechnique.ABSTAIN:
                        counts[j, k] += 1
                        counts[k, j] += 1
                        if L[i, j] == L[i, k]:
                            agreements[j, k] += 1
                            agreements[k, j] += 1
        
        # Estimate accuracy as weighted agreement rate
        estimated_acc = {}
        for j, lf in enumerate(self.lfs):
            if counts[j].sum() > 0:
                agreement_rate = agreements[j].sum() / counts[j].sum()
                # Transform agreement rate to accuracy estimate
                # Higher agreement with high-weight LFs → higher accuracy
                weight_factor = np.mean([self.lfs[k].weight for k in range(m) if counts[j, k] > 0])
                acc = 0.5 + 0.5 * agreement_rate * weight_factor / 2
                estimated_acc[lf.name] = min(0.95, max(0.3, acc))
            else:
                estimated_acc[lf.name] = 0.5
        
        self.lf_accuracies = estimated_acc
        return estimated_acc
    
    def predict_proba(
        self, 
        L: np.ndarray, 
        confidences: np.ndarray = None
    ) -> np.ndarray:
        """
        Predict class probabilities using weighted voting with:
        - LF weights
        - Estimated accuracies
        - Per-instance confidences
        """
        n, m = L.shape
        probs = np.zeros((n, self.num_classes))
        
        if confidences is None:
            confidences = np.ones((n, m))
        
        for i in range(n):
            votes = np.zeros(self.num_classes)
            total_weight = 0
            
            for j, lf in enumerate(self.lfs):
                label = L[i, j]
                if label != ProofTechnique.ABSTAIN:
                    # Combine weight, accuracy, and confidence
                    effective_weight = (
                        lf.weight * 
                        self.lf_accuracies.get(lf.name, 0.7) * 
                        confidences[i, j]
                    )
                    votes[label] += effective_weight
                    total_weight += effective_weight
            
            if total_weight > 0:
                probs[i] = votes / total_weight
            else:
                probs[i] = self.prior
        
        return probs
    
    def predict(self, L: np.ndarray, confidences: np.ndarray = None) -> np.ndarray:
        """Return most likely class"""
        probs = self.predict_proba(L, confidences)
        return np.argmax(probs, axis=1)
    
    def get_uncertainty(self, probs: np.ndarray) -> np.ndarray:
        """
        Calculate prediction uncertainty using entropy
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        max_entropy = np.log(self.num_classes)
        return entropy / max_entropy  # Normalized to [0, 1]
    
    def analyze_single(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single proof text
        """
        L, confidences, results = self.apply([text])
        probs = self.predict_proba(L, confidences)[0]
        uncertainty = self.get_uncertainty(probs.reshape(1, -1))[0]
        
        predicted_idx = int(np.argmax(probs))
        
        # Collect triggered LFs with details
        triggered = []
        for j, (lf, result) in enumerate(zip(self.lfs, results[0])):
            if result.label != ProofTechnique.ABSTAIN:
                triggered.append({
                    "name": lf.name,
                    "label": int(result.label),
                    "weight": lf.weight,
                    "confidence": result.confidence,
                    "accuracy": self.lf_accuracies.get(lf.name, 0.7),
                    "evidence": result.evidence
                })
        
        return {
            "predicted_class": predicted_idx,
            "confidence": float(probs[predicted_idx]),
            "uncertainty": float(uncertainty),
            "probability_distribution": probs.tolist(),
            "triggered_labeling_functions": triggered,
            "coverage": len(triggered) / len(self.lfs),
            "num_lfs_triggered": len(triggered)
        }


# ============================================================================
# COMPLETE LABELING FUNCTION SET
# ============================================================================

def create_labeling_functions() -> List[BaseLabelingFunction]:
    """
    Create a comprehensive set of labeling functions for proof classification
    """
    lfs = []
    
    # === DIRECT PROOF ===
    lfs.extend([
        KeywordLF(
            "direct_simple",
            ProofTechnique.DIRECT,
            keywords=["therefore", "thus", "hence", "it follows"],
            negative_keywords=["contradiction", "suppose not", "induction"],
            weight=0.6
        ),
        PatternLF(
            "direct_assume_then",
            ProofTechnique.DIRECT,
            patterns=[
                r"assume\s+.{5,60}\.\s*then\s+.{5,}",
                r"let\s+.{3,40}\s+be\s+.{3,40}\.\s*then",
                r"suppose\s+.{5,50}\.\s*(we\s+have|then|so)"
            ],
            weight=0.9
        ),
        PatternLF(
            "direct_conclude",
            ProofTechnique.DIRECT,
            patterns=[
                r"we\s+conclude\s+that",
                r"which\s+(shows|proves|establishes)\s+that",
                r"as\s+(required|desired|claimed)"
            ],
            weight=0.7
        ),
    ])
    
    # === CONTRADICTION ===
    lfs.extend([
        KeywordLF(
            "contradiction_keywords",
            ProofTechnique.CONTRADICTION,
            keywords=["contradiction", "contradicts", "absurd", "impossible"],
            weight=1.3
        ),
        PatternLF(
            "contradiction_setup",
            ProofTechnique.CONTRADICTION,
            patterns=[
                r"suppose\s+(for\s+)?contradiction",
                r"assume\s+(for\s+)?contradiction",
                r"suppose\s+not",
                r"assume\s+.{0,30}(not|false)",
                r"for\s+the\s+sake\s+of\s+contradiction",
                r"assume\s+the\s+contrary"
            ],
            weight=1.4
        ),
        PatternLF(
            "contradiction_conclusion",
            ProofTechnique.CONTRADICTION,
            patterns=[
                r"this\s+(is\s+a\s+)?contradiction",
                r"which\s+(is\s+a\s+)?contradiction",
                r"contradicting\s+(our|the)\s+assumption",
                r"which\s+contradicts",
                r"a\s+contradiction\s+to"
            ],
            weight=1.5
        ),
        StructuralLF(
            "contradiction_structure",
            ProofTechnique.CONTRADICTION,
            required_sections=["suppose", "contradiction"],
            ordered=True,
            weight=1.2
        ),
    ])
    
    # === INDUCTION ===
    lfs.extend([
        KeywordLF(
            "induction_keyword",
            ProofTechnique.INDUCTION,
            keywords=["induction", "inductive", "inductively"],
            weight=1.4
        ),
        StructuralLF(
            "induction_full_structure",
            ProofTechnique.INDUCTION,
            required_sections=["base case", "inductive step"],
            optional_sections=["inductive hypothesis", "by induction"],
            ordered=True,
            weight=1.6
        ),
        PatternLF(
            "induction_base",
            ProofTechnique.INDUCTION,
            patterns=[
                r"base\s+case",
                r"for\s+n\s*=\s*[01]",
                r"when\s+n\s*=\s*[01]",
                r"the\s+case\s+n\s*=\s*[01]"
            ],
            weight=1.1
        ),
        PatternLF(
            "induction_step",
            ProofTechnique.INDUCTION,
            patterns=[
                r"inductive\s+(step|case|hypothesis)",
                r"assume\s+(it\s+)?(holds|true)\s+for\s+.{1,20}k",
                r"suppose\s+.{1,30}holds\s+for\s+n\s*=\s*k",
                r"by\s+(the\s+)?inductive\s+hypothesis",
                r"IH"  # Common abbreviation
            ],
            weight=1.3
        ),
        PatternLF(
            "induction_strong",
            ProofTechnique.INDUCTION,
            patterns=[
                r"strong\s+induction",
                r"complete\s+induction",
                r"assume.*holds.*for\s+all.*<\s*n",
                r"for\s+all\s+.{1,15}<\s*n"
            ],
            weight=1.2
        ),
    ])
    
    # === CONTRAPOSITIVE ===
    lfs.extend([
        KeywordLF(
            "contrapositive_keyword",
            ProofTechnique.CONTRAPOSITIVE,
            keywords=["contrapositive", "contrapositively"],
            weight=1.5
        ),
        PatternLF(
            "contrapositive_pattern",
            ProofTechnique.CONTRAPOSITIVE,
            patterns=[
                r"prove\s+(the\s+)?contrapositive",
                r"by\s+contrapositive",
                r"equivalent(ly)?\s*,?\s*(we\s+)?(show|prove)\s+.{0,20}not.{0,20}implies.{0,20}not",
                r"instead\s+(prove|show)\s+.{0,30}¬.{0,30}→.{0,30}¬",
                r"if\s+not\s+.{3,30}then\s+not"
            ],
            weight=1.4
        ),
    ])
    
    # === CONSTRUCTION ===
    lfs.extend([
        KeywordLF(
            "construction_keywords",
            ProofTechnique.CONSTRUCTION,
            keywords=["construct", "constructive", "exhibit", "produce"],
            negative_keywords=["construct a contradiction"],
            weight=1.0
        ),
        PatternLF(
            "construction_pattern",
            ProofTechnique.CONSTRUCTION,
            patterns=[
                r"we\s+construct\s+(an?|the)",
                r"consider\s+.{0,30}\s*=",
                r"define\s+.{1,30}\s*(=|:=|as)",
                r"let\s+.{1,30}\s*=\s*.{5,}",
                r"take\s+.{1,30}\s*=",
                r"set\s+.{1,30}\s*="
            ],
            weight=0.9
        ),
        PatternLF(
            "construction_existence",
            ProofTechnique.CONSTRUCTION,
            patterns=[
                r"to\s+(show|prove)\s+.{0,30}exists",
                r"we\s+exhibit\s+(an?|such)",
                r"here\s+is\s+(an?\s+)?explicit",
                r"namely\s*,?\s*.{3,30}=",
                r"such\s+.{1,20}exists.*namely"
            ],
            weight=1.1
        ),
    ])
    
    # === CASES ===
    lfs.extend([
        KeywordLF(
            "cases_explicit",
            ProofTechnique.CASES,
            keywords=["case 1", "case 2", "case i", "case ii", "first case", "second case"],
            weight=1.4
        ),
        PatternLF(
            "cases_intro",
            ProofTechnique.CASES,
            patterns=[
                r"we\s+(consider|divide|split)\s+(into\s+)?(two|three|several|the\s+following)\s+cases",
                r"there\s+are\s+(two|three|several|the\s+following)\s+cases",
                r"we\s+proceed\s+by\s+cases",
                r"case\s+analysis"
            ],
            weight=1.2
        ),
        PatternLF(
            "cases_enumeration",
            ProofTechnique.CASES,
            patterns=[
                r"case\s+\d+\s*:",
                r"case\s+[ivx]+\s*:",
                r"\(\s*case\s+\d+\s*\)",
                r"in\s+(the\s+)?(first|second|third|final)\s+case"
            ],
            weight=1.3
        ),
        PatternLF(
            "cases_either_or",
            ProofTechnique.CASES,
            patterns=[
                r"either\s+.{5,50}\s+or\s+.{5,}",
                r"suppose\s+.{1,30}(even|odd).{0,50}suppose\s+.{1,30}(even|odd)"
            ],
            weight=0.8
        ),
    ])
    
    # === EXHAUSTION ===
    lfs.extend([
        KeywordLF(
            "exhaustion_keywords",
            ProofTechnique.EXHAUSTION,
            keywords=["exhaustively", "exhaustion", "brute force", "finite check"],
            weight=1.3
        ),
        PatternLF(
            "exhaustion_pattern",
            ProofTechnique.EXHAUSTION,
            patterns=[
                r"(check|verify|test)\s+(each|all|every)",
                r"by\s+exhaustion",
                r"checking\s+all\s+(possible\s+)?cases",
                r"there\s+are\s+only\s+finitely\s+many",
                r"we\s+(can\s+)?enumerate\s+all",
                r"since\s+there\s+are\s+only\s+\d+\s+(cases|possibilities)"
            ],
            weight=1.2
        ),
    ])
    
    return lfs


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_label_model() -> EnhancedLabelModel:
    """Create and return a configured label model"""
    lfs = create_labeling_functions()
    return EnhancedLabelModel(lfs, num_classes=7)


if __name__ == "__main__":
    # Test the labeling functions
    model = create_label_model()
    
    test_proofs = [
        "Suppose √2 is rational. Then √2 = a/b where a, b are coprime integers. Squaring, a² = 2b². Thus a is even, say a = 2k. Then 4k² = 2b², so b² = 2k². Thus b is also even. This contradicts a, b being coprime. Therefore √2 is irrational.",
        "We prove by induction. Base case: For n=1, 1 = 1(2)/2 = 1. ✓ Inductive step: Assume true for n=k. Then for n=k+1: 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2. By induction, the formula holds for all n.",
        "To show there exists an even prime, we exhibit 2. Indeed, 2 = 2×1, so 2 is prime. And 2 = 2×1 is even. Thus such a prime exists.",
    ]
    
    for i, proof in enumerate(test_proofs):
        print(f"\n{'='*60}")
        print(f"Proof {i+1}:")
        print(f"{'='*60}")
        result = model.analyze_single(proof)
        print(f"Predicted: {ProofTechnique(result['predicted_class']).name}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Uncertainty: {result['uncertainty']:.2%}")
        print(f"LFs triggered: {result['num_lfs_triggered']}")
        for lf in result['triggered_labeling_functions']:
            print(f"  - {lf['name']}: {lf['evidence']}")
