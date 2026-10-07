"""
Tests for src/demo_data.py (built-in Demo Mode inbox). Runs fully offline
and never touches Gmail or the SQLite cache.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.demo_data import get_demo_emails
from src.predict import classify_email, models_available

REQUIRED_CATEGORIES = {"Education", "Work", "Finance", "Shopping", "Security", "Personal", "Promotions", "Social"}


def test_demo_inbox_size_and_fields():
    emails = get_demo_emails()
    assert 30 <= len(emails) <= 50
    for e in emails:
        for field in ("message_id", "sender_name", "sender_email", "subject", "body", "date",
                      "is_read", "is_starred", "expected_category", "expected_priority", "expected_score"):
            assert e[field] is not None and e[field] != ""
        assert 0 <= e["expected_score"] <= 100
        assert e["expected_priority"] in {"HIGH", "MEDIUM", "LOW"}


def test_demo_inbox_covers_required_categories_and_states():
    emails = get_demo_emails()
    assert REQUIRED_CATEGORIES <= {e["expected_category"] for e in emails}
    assert any(not e["is_read"] for e in emails) and any(e["is_read"] for e in emails)
    assert any(e["is_starred"] for e in emails)


def test_demo_message_ids_are_unique():
    ids = [e["message_id"] for e in get_demo_emails()]
    assert len(ids) == len(set(ids))


@pytest.mark.skipif(not models_available(), reason="Models not trained - run python src/train_models.py")
def test_ml_models_classify_demo_inbox_reasonably():
    emails = get_demo_emails()
    correct = sum(classify_email(e["subject"], e["body"])["category"] == e["expected_category"] for e in emails)
    assert correct / len(emails) >= 0.8
