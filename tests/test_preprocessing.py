"""Tests for src/preprocessing.py. Does not require Gmail access."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    clean_text,
    strip_html,
    combine_subject_body,
    preprocess_for_vectorizer,
    extract_metadata_features,
    count_keyword_hits,
)


def test_clean_text_lowercases_and_strips_punctuation():
    result = clean_text("URGENT!! Your Account Needs Attention.")
    assert result == "urgent your account needs attention"


def test_clean_text_removes_urls():
    result = clean_text("Visit https://example.com/offer now")
    assert "http" not in result
    assert "example.com" not in result


def test_clean_text_removes_email_addresses():
    result = clean_text("Contact me at john.doe@example.com for info")
    assert "@" not in result


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_strip_html_removes_tags():
    html = "<html><body><p>Hello <b>World</b></p><script>alert(1)</script></body></html>"
    result = strip_html(html)
    assert "<" not in result
    assert "alert" not in result
    assert "Hello" in result and "World" in result


def test_combine_subject_body_weights_subject():
    combined = combine_subject_body("Subject Text", "Body Text")
    assert combined.count("Subject Text") == 2
    assert "Body Text" in combined


def test_preprocess_for_vectorizer_is_clean_and_combined():
    result = preprocess_for_vectorizer("Urgent!", "Please respond ASAP.")
    assert result == result.lower()
    assert "!" not in result


def test_extract_metadata_features_detects_security_terms():
    features = extract_metadata_features("New login detected", "Please verify your password immediately.")
    assert features["security_hits"] > 0
    assert features["urgency_hits"] > 0
    assert features["subject_length"] == len("New login detected")


def test_extract_metadata_features_no_signals():
    features = extract_metadata_features("Weekly newsletter", "Here are this week's top stories.")
    assert features["security_hits"] == 0
    assert features["urgency_hits"] == 0


def test_count_keyword_hits():
    assert count_keyword_hits("this is urgent and critical", ["urgent", "critical", "alert"]) == 2
    assert count_keyword_hits("", ["urgent"]) == 0
    assert count_keyword_hits(None, ["urgent"]) == 0
