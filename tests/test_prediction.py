"""
Tests for src/predict.py (category + priority ML prediction).

These tests require the models to already be trained:
    python src/train_models.py

They do NOT require any Gmail account access.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.predict import (
    classify_email,
    predict_category,
    predict_priority,
    compute_priority_score,
    models_available,
)

pytestmark = pytest.mark.skipif(
    not models_available(),
    reason="ML models not trained yet - run 'python src/train_models.py' first",
)


def test_predict_category_returns_valid_label():
    category, confidence = predict_category(
        "Your order has been shipped", "Your package is on its way and will arrive soon."
    )
    assert isinstance(category, str)
    assert len(category) > 0
    assert confidence is None or 0.0 <= confidence <= 1.0


def test_predict_priority_returns_valid_label():
    priority, confidence, proba, meta = predict_priority(
        "New login detected", "We detected a suspicious login, please verify your password immediately.", "Security"
    )
    assert priority in ["HIGH", "MEDIUM", "LOW"]
    assert confidence is None or 0.0 <= confidence <= 1.0
    assert isinstance(meta, dict)


def test_security_email_classified_as_security_category():
    category, _ = predict_category(
        "New login detected on your account",
        "We detected a new login from an unrecognized device. Please verify your password immediately.",
    )
    assert category == "Security"


def test_newsletter_email_gets_low_priority_score():
    result = classify_email(
        "Weekly technology newsletter",
        "Here is your weekly roundup of the top technology stories, curated just for you.",
    )
    assert result["priority_score"] <= 50


def test_security_email_gets_high_priority_score():
    result = classify_email(
        "Urgent security alert: verify your password immediately",
        "Suspicious login detected. Your account will be locked unless you verify immediately.",
    )
    assert result["priority_score"] >= 50


def test_compute_priority_score_breakdown_structure():
    breakdown = compute_priority_score("HIGH", {"LOW": 0.1, "MEDIUM": 0.2, "HIGH": 0.7}, {
        "urgency_hits": 1, "deadline_hits": 0, "security_hits": 0,
    })
    assert "ml_base_score" in breakdown
    assert "rule_adjustment" in breakdown
    assert "final_score" in breakdown
    assert 0 <= breakdown["final_score"] <= 100


def test_classify_email_returns_complete_result():
    result = classify_email("Meeting with family this Sunday", "Let's all get together for lunch this Sunday.")
    assert "category" in result
    assert "priority" in result
    assert "priority_score" in result
    assert "explanation" in result
    assert isinstance(result["explanation"], list)
    assert len(result["explanation"]) > 0
