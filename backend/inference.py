"""
Inference Module for Neural Proof Assistant

Loads the trained model and provides prediction functions.
This module handles:
- Loading pre-trained sentence transformer
- Loading the trained classifier
- Running inference on new proofs
"""

import pickle
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Lazy loading for sentence transformers (heavy import)
_encoder = None
_classifier = None
_label_encoder = None
_config = None

MODELS_DIR = Path(__file__).parent / "models"

# Technique metadata
TECHNIQUE_INFO = {
    "Direct Proof": {
        "description": "Proves P→Q by assuming P and deriving Q through logical steps",
        "color": "#10B981",
        "icon": "→"
    },
    "Proof by Contradiction": {
        "description": "Assumes ¬Q and derives a contradiction, proving Q must be true",
        "color": "#EF4444",
        "icon": "⊥"
    },
    "Mathematical Induction": {
        "description": "Proves base case P(0), then P(n)→P(n+1) for all n",
        "color": "#8B5CF6",
        "icon": "∀n"
    },
    "Contrapositive": {
        "description": "Proves P→Q by proving ¬Q→¬P",
        "color": "#F59E0B",
        "icon": "¬"
    },
    "Proof by Construction": {
        "description": "Proves existence by explicitly constructing an example",
        "color": "#06B6D4",
        "icon": "∃"
    },
    "Proof by Cases": {
        "description": "Divides the proof into exhaustive cases and proves each",
        "color": "#EC4899",
        "icon": "∨"
    },
    "Proof by Exhaustion": {
        "description": "Verifies all finite possibilities",
        "color": "#84CC16",
        "icon": "∀"
    }
}


def get_encoder():
    """Lazy load the sentence transformer encoder"""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        model_name = get_config().get("model_name", "all-MiniLM-L6-v2")
        print(f"Loading sentence transformer: {model_name}")
        _encoder = SentenceTransformer(model_name)
        print("✅ Encoder loaded")
    return _encoder


def get_classifier():
    """Load the trained classifier"""
    global _classifier
    if _classifier is None:
        classifier_path = MODELS_DIR / "classifier.pkl"
        if not classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier not found at {classifier_path}. "
                "Please run the training notebook first."
            )
        with open(classifier_path, 'rb') as f:
            _classifier = pickle.load(f)
        print("✅ Classifier loaded")
    return _classifier


def get_label_encoder():
    """Load the label encoder"""
    global _label_encoder
    if _label_encoder is None:
        encoder_path = MODELS_DIR / "label_encoder.pkl"
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at {encoder_path}. "
                "Please run the training notebook first."
            )
        with open(encoder_path, 'rb') as f:
            _label_encoder = pickle.load(f)
        print("✅ Label encoder loaded")
    return _label_encoder


def get_config() -> dict:
    """Load model configuration"""
    global _config
    if _config is None:
        config_path = MODELS_DIR / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                _config = json.load(f)
        else:
            _config = {
                "model_name": "all-MiniLM-L6-v2",
                "embedding_dim": 384,
                "num_classes": 7
            }
    return _config


def is_model_available() -> bool:
    """Check if trained model files exist"""
    return (MODELS_DIR / "classifier.pkl").exists() and \
           (MODELS_DIR / "label_encoder.pkl").exists()


def predict(text: str) -> Dict[str, Any]:
    """
    Predict the proof technique for a given text.
    
    Args:
        text: The proof text to classify
        
    Returns:
        Dictionary with prediction results
    """
    encoder = get_encoder()
    classifier = get_classifier()
    label_enc = get_label_encoder()
    
    # Generate embedding
    embedding = encoder.encode([text])
    
    # Get prediction and probabilities
    pred_encoded = classifier.predict(embedding)[0]
    pred_proba = classifier.predict_proba(embedding)[0]
    
    # Decode label
    pred_label = label_enc.inverse_transform([pred_encoded])[0]
    technique_name = get_config()["classes"][pred_encoded]
    confidence = float(pred_proba[pred_encoded])
    
    # Build probability distribution
    prob_dist = {}
    for i, prob in enumerate(pred_proba):
        name = get_config()["classes"][i]
        info = TECHNIQUE_INFO.get(name, {})
        prob_dist[name] = {
            "probability": float(prob),
            "color": info.get("color", "#6B7280"),
            "icon": info.get("icon", "?"),
            "description": info.get("description", "")
        }
    
    # Get technique info
    tech_info = TECHNIQUE_INFO.get(technique_name, {})
    
    return {
        "predicted_technique": technique_name,
        "confidence": confidence,
        "technique_info": {
            "name": technique_name,
            "description": tech_info.get("description", ""),
            "color": tech_info.get("color", "#6B7280"),
            "icon": tech_info.get("icon", "?")
        },
        "probability_distribution": prob_dist,
        "model_type": "neural",
        "embedding_dim": get_config().get("embedding_dim", 384)
    }


def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Predict proof techniques for multiple texts.
    
    Args:
        texts: List of proof texts
        
    Returns:
        List of prediction results
    """
    encoder = get_encoder()
    classifier = get_classifier()
    label_enc = get_label_encoder()
    
    # Generate embeddings for all texts
    embeddings = encoder.encode(texts, show_progress_bar=len(texts) > 10)
    
    # Get predictions
    pred_encoded = classifier.predict(embeddings)
    pred_proba = classifier.predict_proba(embeddings)
    
    results = []
    for i, text in enumerate(texts):
        technique_name = get_config()["classes"][pred_encoded[i]]
        confidence = float(pred_proba[i][pred_encoded[i]])
        tech_info = TECHNIQUE_INFO.get(technique_name, {})
        
        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "predicted_technique": technique_name,
            "confidence": confidence,
            "technique_info": {
                "name": technique_name,
                "color": tech_info.get("color", "#6B7280"),
                "icon": tech_info.get("icon", "?")
            }
        })
    
    return results


def get_model_info() -> Dict[str, Any]:
    """Return information about the loaded model"""
    config = get_config()
    return {
        "model_available": is_model_available(),
        "encoder_model": config.get("model_name", "unknown"),
        "classifier_type": config.get("classifier_type", "unknown"),
        "embedding_dim": config.get("embedding_dim", 0),
        "num_classes": config.get("num_classes", 0),
        "classes": config.get("classes", []),
        "training_samples": config.get("training_samples", 0),
        "cv_accuracy": config.get("cv_accuracy", 0)
    }


if __name__ == "__main__":
    # Test the inference module
    print("Testing inference module...\n")
    
    if not is_model_available():
        print("❌ Model files not found. Please run the training notebook first.")
        print(f"   Expected location: {MODELS_DIR}")
    else:
        test_proof = """
        Suppose √2 is rational. Then √2 = a/b where a, b have no common factors.
        Squaring, a² = 2b². This means a is even. Write a = 2k.
        Then b² = 2k², so b is also even. This contradicts a/b being in lowest terms.
        Therefore √2 is irrational.
        """
        
        result = predict(test_proof)
        print(f"Predicted: {result['predicted_technique']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print("\n✅ Inference module working!")
