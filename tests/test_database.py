"""
Tests for src/database.py (SQLite cache). Uses a temporary database file so
the project's real data/inboxiq.db is never touched. Does not require Gmail.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src import database as db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the database module at a throwaway SQLite file for this test only."""
    temp_path = tmp_path / "test_inboxiq.db"
    monkeypatch.setattr(db, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(db, "DB_PATH", str(temp_path))
    db.init_db()
    yield db
    db.clear_all_data()


def sample_record(message_id="msg-1", category="Work", priority="HIGH", score=90):
    return {
        "message_id": message_id,
        "thread_id": "thread-1",
        "sender": "boss@example.com",
        "recipient": "me@example.com",
        "subject": "Test subject",
        "body": "Test body content",
        "snippet": "Test body",
        "date": "2026-01-01T10:00:00",
        "labels": "INBOX",
        "category": category,
        "priority": priority,
        "priority_score": score,
        "confidence": 0.9,
    }


def test_upsert_and_retrieve_email(temp_db):
    temp_db.upsert_email(sample_record())
    fetched = temp_db.get_email_by_id("msg-1")
    assert fetched is not None
    assert fetched["subject"] == "Test subject"
    assert fetched["category"] == "Work"
    assert fetched["priority"] == "HIGH"


def test_upsert_is_idempotent_for_same_message_id(temp_db):
    temp_db.upsert_email(sample_record(message_id="msg-2", category="Work"))
    temp_db.upsert_email(sample_record(message_id="msg-2", category="Finance"))  # update, not duplicate

    all_emails = temp_db.get_all_emails()
    matching = [e for e in all_emails if e["message_id"] == "msg-2"]
    assert len(matching) == 1
    assert matching[0]["category"] == "Finance"


def test_get_existing_message_ids(temp_db):
    temp_db.upsert_email(sample_record(message_id="msg-3"))
    temp_db.upsert_email(sample_record(message_id="msg-4"))
    ids = temp_db.get_existing_message_ids()
    assert ids == {"msg-3", "msg-4"}


def test_filter_by_category_and_priority(temp_db):
    temp_db.upsert_email(sample_record(message_id="msg-5", category="Security", priority="HIGH"))
    temp_db.upsert_email(sample_record(message_id="msg-6", category="Promotions", priority="LOW"))

    high_only = temp_db.get_all_emails(priority="HIGH")
    assert len(high_only) == 1
    assert high_only[0]["message_id"] == "msg-5"

    security_only = temp_db.get_all_emails(category="Security")
    assert len(security_only) == 1
    assert security_only[0]["message_id"] == "msg-5"


def test_search_by_subject(temp_db):
    record = sample_record(message_id="msg-7")
    record["subject"] = "Unique Searchable Subject"
    temp_db.upsert_email(record)

    results = temp_db.get_all_emails(search="Searchable")
    assert len(results) == 1
    assert results[0]["message_id"] == "msg-7"


def test_get_stats(temp_db):
    temp_db.upsert_email(sample_record(message_id="msg-8", priority="HIGH"))
    temp_db.upsert_email(sample_record(message_id="msg-9", priority="LOW"))

    stats = temp_db.get_stats()
    assert stats["total_emails"] == 2
    assert stats["high_priority"] == 1
    assert stats["low_priority"] == 1


def test_clear_all_data(temp_db):
    temp_db.upsert_email(sample_record(message_id="msg-10"))
    assert temp_db.get_stats()["total_emails"] == 1

    temp_db.clear_all_data()
    assert temp_db.get_stats()["total_emails"] == 0
