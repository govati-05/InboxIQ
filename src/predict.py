"""
InboxIQ - Live Prediction Pipeline
=====================================
Loads the trained category + priority models and exposes a single
classify_email() function used by both Gmail sync and the manual
"Analyze Email" page.

Pipeline per email:
    subject/body -> preprocess -> TF-IDF -> category model -> category
                 -> (text + metadata + category) -> priority model -> priority
                 -> priority probabilities -> 0-100 priority score
"""

import os
import json

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
import pandas as pd

from src.preprocessing import preprocess_for_vectorizer, extract_metadata_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CATEGORY_MODEL_PATH = os.path.join(MODELS_DIR, "category_model.pkl")
CATEGORY_VECTORIZER_PATH = os.path.join(MODELS_DIR, "category_vectorizer.pkl")
PRIORITY_MODEL_PATH = os.path.join(MODELS_DIR, "priority_model.pkl")
PRIORITY_VECTORIZER_PATH = os.path.join(MODELS_DIR, "priority_vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Anchor scores used to convert priority class probabilities into a smooth
# 0-100 score. This is the "ML prediction" portion of the priority score.
PRIORITY_ANCHORS = {"LOW": 15, "MEDIUM": 50, "HIGH": 85}

# Maximum total points the deterministic business-rule layer may add or
# subtract on top of the ML-derived base score. Kept deliberately small so
# the ML model's prediction remains the dominant signal.
MAX_RULE_ADJUSTMENT = 10


class ModelsNotTrainedError(Exception):
    """Raised when model files are missing, so the UI can show a clear fix-it message."""
    pass


_cache = {}


def _require_file(path, what):
    if not os.path.exists(path):
        raise ModelsNotTrainedError(
            f"{what} not found at {os.path.relpath(path, PROJECT_ROOT)}. "
            f"Please run 'python src/train_models.py' first to train and save the models."
        )


def load_models():
    """Load and cache all model artifacts. Raises ModelsNotTrainedError with a friendly message if missing."""
    if _cache:
        return _cache

    _require_file(CATEGORY_MODEL_PATH, "Category model")
    _require_file(CATEGORY_VECTORIZER_PATH, "Category vectorizer")
    _require_file(PRIORITY_MODEL_PATH, "Priority model")
    _require_file(PRIORITY_VECTORIZER_PATH, "Priority vectorizer")

    _cache["category_model"] = joblib.load(CATEGORY_MODEL_PATH)
    _cache["category_vectorizer"] = joblib.load(CATEGORY_VECTORIZER_PATH)
    _cache["priority_model"] = joblib.load(PRIORITY_MODEL_PATH)
    _cache["priority_bundle"] = joblib.load(PRIORITY_VECTORIZER_PATH)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            _cache["metrics"] = json.load(f)
    else:
        _cache["metrics"] = None

    return _cache


def models_available():
    return all(os.path.exists(p) for p in [
        CATEGORY_MODEL_PATH, CATEGORY_VECTORIZER_PATH, PRIORITY_MODEL_PATH, PRIORITY_VECTORIZER_PATH
    ])


def get_metrics():
    models = load_models()
    return models.get("metrics")


def predict_category(subject, body):
    """Returns (category_label, confidence_or_None)."""
    models = load_models()
    clean = preprocess_for_vectorizer(subject, body)
    vec = models["category_vectorizer"].transform([clean])
    model = models["category_model"]

    prediction = model.predict(vec)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = float(np.max(proba))
    return prediction, confidence


def _build_priority_feature_vector(subject, body, category):
    models = load_models()
    bundle = models["priority_bundle"]
    tfidf = bundle["tfidf"]
    scaler = bundle["scaler"]
    categories = bundle["categories"]
    meta_cols = bundle["meta_columns"]

    clean = preprocess_for_vectorizer(subject, body)
    text_vec = tfidf.transform([clean])

    meta = extract_metadata_features(subject, body, category)
    meta_row = pd.DataFrame([{col: meta.get(col, 0) for col in meta_cols}])
    meta_scaled = scaler.transform(meta_row.values)

    category_onehot = np.zeros((1, len(categories)))
    if category in categories:
        category_onehot[0, categories.index(category)] = 1.0

    combined = hstack([text_vec, csr_matrix(meta_scaled), csr_matrix(category_onehot)]).tocsr()
    return combined, meta


def predict_priority(subject, body, category):
    """Returns (priority_label, confidence_or_None, class_probabilities_dict_or_None, metadata_features)."""
    models = load_models()
    model = models["priority_model"]

    feature_vec, meta = _build_priority_feature_vector(subject, body, category)
    prediction = model.predict(feature_vec)[0]

    confidence = None
    proba_dict = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feature_vec)[0]
        classes = list(model.classes_)
        proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
        confidence = float(max(proba))

    return prediction, confidence, proba_dict, meta


def compute_priority_score(priority_label, proba_dict, meta_features):
    """
    Compute a transparent 0-100 priority score, clearly split into:
        ml_base_score      - derived purely from the model's class probabilities
        rule_adjustment     - a small, bounded, deterministic nudge based on
                               engineered signal counts (urgency/deadline/security)
        final_score         - ml_base_score + rule_adjustment, clipped to [0, 100]
    """
    if proba_dict:
        ml_base_score = sum(proba_dict.get(label, 0.0) * anchor for label, anchor in PRIORITY_ANCHORS.items())
    else:
        # No probabilities available (model has no predict_proba) - fall back
        # to the anchor for the predicted class alone, clearly documented.
        ml_base_score = PRIORITY_ANCHORS.get(priority_label, 50)

    signal_count = (
        meta_features.get("urgency_hits", 0)
        + meta_features.get("deadline_hits", 0)
        + meta_features.get("security_hits", 0)
    )
    # Each detected signal nudges the score slightly; capped at MAX_RULE_ADJUSTMENT.
    rule_adjustment = min(signal_count * 2, MAX_RULE_ADJUSTMENT)

    final_score = int(round(min(max(ml_base_score + rule_adjustment, 0), 100)))

    return {
        "ml_base_score": round(ml_base_score, 1),
        "rule_adjustment": rule_adjustment,
        "final_score": final_score,
    }


def build_explanation(category, category_confidence, priority, score_breakdown, meta_features):
    """Generate a simple, truthful explanation list based on actual features/predictions."""
    reasons = []

    if meta_features.get("security_hits", 0) > 0:
        reasons.append("Security-related language detected in the email.")
    if meta_features.get("urgency_hits", 0) > 0:
        reasons.append("Urgency-indicating words detected (e.g. 'urgent', 'immediately', 'asap').")
    if meta_features.get("deadline_hits", 0) > 0:
        reasons.append("Deadline/time-sensitive language detected.")

    if priority == "HIGH":
        reasons.append(f"The priority model classified this email as HIGH priority.")
    elif priority == "MEDIUM":
        reasons.append(f"The priority model classified this email as MEDIUM priority.")
    else:
        reasons.append(f"The priority model classified this email as LOW priority.")

    if category_confidence is not None:
        reasons.append(f"The category model predicted '{category}' with {category_confidence * 100:.0f}% confidence.")
    else:
        reasons.append(f"The category model predicted '{category}' (confidence unavailable for this model).")

    reasons.append(
        f"Priority score = ML-derived base score ({score_breakdown['ml_base_score']}) "
        f"+ rule-based adjustment ({score_breakdown['rule_adjustment']:+d}) = {score_breakdown['final_score']}/100."
    )

    return reasons


def classify_email(subject, body):
    """
    Full end-to-end classification used for both Gmail sync and manual
    analysis. Returns a dict ready to store in SQLite / show in the UI.
    """
    category, category_confidence = predict_category(subject, body)
    priority, priority_confidence, proba_dict, meta = predict_priority(subject, body, category)
    score_breakdown = compute_priority_score(priority, proba_dict, meta)
    explanation = build_explanation(category, category_confidence, priority, score_breakdown, meta)

    return {
        "category": category,
        "category_confidence": category_confidence,
        "priority": priority,
        "priority_confidence": priority_confidence,
        "priority_score": score_breakdown["final_score"],
        "score_breakdown": score_breakdown,
        "explanation": explanation,
        "metadata_features": meta,
    }
