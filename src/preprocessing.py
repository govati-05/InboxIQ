"""
InboxIQ - Text Preprocessing Module
=====================================
Reusable text cleaning and feature extraction utilities shared by the
training pipeline (src/train_models.py) and the live prediction pipeline
(src/predict.py). Keeping this logic in one place guarantees that emails
are preprocessed the exact same way at train time and at prediction time.
"""

import re
from html import unescape

from bs4 import BeautifulSoup

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z0-9\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Keyword lists used for engineered features (urgency / deadline / security
# signals) that are combined with TF-IDF text features for priority
# prediction. These are NOT used to directly assign priority labels; they
# are inputs into a trained ML model, not a rule engine.
URGENCY_WORDS = [
    "urgent", "immediately", "asap", "action required", "important",
    "critical", "alert", "warning", "now", "deadline", "final notice",
    "last chance", "expire", "expires", "expiring", "hurry",
]

DEADLINE_WORDS = [
    "due", "deadline", "by tomorrow", "today", "tonight", "within 24 hours",
    "expires", "last date", "before", "submit by",
]

SECURITY_WORDS = [
    "password", "login", "verify", "otp", "security", "suspicious",
    "unauthorized", "breach", "authentication", "2fa", "two-factor",
    "verification code", "locked", "blocked",
]


def strip_html(text):
    """Safely extract readable text from an HTML email body (no script execution)."""
    if not text:
        return ""
    try:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ")
    except Exception:
        return text


def clean_text(text):
    """Lowercase, strip HTML/URLs/emails/punctuation, and normalize whitespace."""
    if text is None:
        return ""
    text = unescape(str(text))
    text = strip_html(text)
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text


def combine_subject_body(subject, body):
    """Combine subject + body into a single text blob for vectorization."""
    subject = subject or ""
    body = body or ""
    return f"{subject} {subject} {body}"  # subject weighted x2 (it's usually more informative)


def preprocess_for_vectorizer(subject, body):
    """Full pipeline: combine then clean. This is what TfidfVectorizer consumes."""
    combined = combine_subject_body(subject, body)
    return clean_text(combined)


def count_keyword_hits(text, keywords):
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def extract_metadata_features(subject, body, category=None):
    """
    Extract numeric engineered features used alongside TF-IDF for the
    priority model: lengths + keyword-hit counts. Returns a plain dict so
    callers can build a DataFrame/array consistently at train and predict
    time.
    """
    subject = subject or ""
    body = body or ""
    combined_raw = f"{subject} {body}"

    return {
        "subject_length": len(subject),
        "body_length": len(body),
        "urgency_hits": count_keyword_hits(combined_raw, URGENCY_WORDS),
        "deadline_hits": count_keyword_hits(combined_raw, DEADLINE_WORDS),
        "security_hits": count_keyword_hits(combined_raw, SECURITY_WORDS),
        "has_exclamation": 1 if "!" in combined_raw else 0,
        "has_numbers": 1 if re.search(r"\d", combined_raw) else 0,
    }
