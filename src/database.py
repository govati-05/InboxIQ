"""
InboxIQ - SQLite Local Cache
==============================
Stores fetched Gmail metadata + ML predictions locally so that InboxIQ does
not need to repeatedly re-download and re-classify the same emails every
time the Streamlit app reruns.

This module ONLY manages InboxIQ's own local cache database
(data/inboxiq.db). It never touches the user's real Gmail account -
clear_all_data() deletes local rows only.
"""

import os
import sqlite3
from datetime import datetime, timezone

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "inboxiq.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS emails (
    message_id TEXT PRIMARY KEY,
    thread_id TEXT,
    sender TEXT,
    recipient TEXT,
    subject TEXT,
    body TEXT,
    snippet TEXT,
    date TEXT,
    labels TEXT,
    category TEXT,
    priority TEXT,
    priority_score INTEGER,
    confidence REAL,
    last_synced TEXT
)
"""

CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_emails_date ON emails(date)"


def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the emails table if it doesn't already exist. Safe to call repeatedly."""
    conn = get_connection()
    try:
        conn.execute(CREATE_TABLE_SQL)
        conn.execute(CREATE_INDEX_SQL)
        conn.commit()
    finally:
        conn.close()


def get_existing_message_ids():
    """Return the set of message_ids already cached, so Gmail sync can skip them."""
    conn = get_connection()
    try:
        cur = conn.execute("SELECT message_id FROM emails")
        return {row["message_id"] for row in cur.fetchall()}
    finally:
        conn.close()


def upsert_email(record):
    """
    Insert a new email record or update it if message_id already exists.
    `record` is a dict matching the table columns (missing keys default to None).
    """
    conn = get_connection()
    try:
        record = dict(record)
        record.setdefault("last_synced", datetime.now(timezone.utc).isoformat())
        columns = [
            "message_id", "thread_id", "sender", "recipient", "subject", "body",
            "snippet", "date", "labels", "category", "priority", "priority_score",
            "confidence", "last_synced",
        ]
        values = [record.get(col) for col in columns]
        placeholders = ", ".join(["?"] * len(columns))
        update_clause = ", ".join(f"{col}=excluded.{col}" for col in columns if col != "message_id")
        sql = f"""
            INSERT INTO emails ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(message_id) DO UPDATE SET {update_clause}
        """
        conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()


def bulk_upsert_emails(records):
    for record in records:
        upsert_email(record)


def get_all_emails(category=None, priority=None, search=None, sort_by="priority"):
    """
    Fetch cached emails with optional category/priority filters and a free-text
    search across subject/sender/body. `sort_by` controls ordering.
    """
    conn = get_connection()
    try:
        query = "SELECT * FROM emails WHERE 1=1"
        params = []

        if category and category != "All":
            query += " AND category = ?"
            params.append(category)

        if priority and priority != "All":
            query += " AND priority = ?"
            params.append(priority.upper())

        if search:
            query += " AND (subject LIKE ? OR sender LIKE ? OR body LIKE ?)"
            like = f"%{search}%"
            params.extend([like, like, like])

        if sort_by == "priority":
            query += """ ORDER BY
                CASE priority WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 WHEN 'LOW' THEN 2 ELSE 3 END,
                date DESC
            """
        elif sort_by == "newest":
            query += " ORDER BY date DESC"
        elif sort_by == "oldest":
            query += " ORDER BY date ASC"
        elif sort_by == "category":
            query += " ORDER BY category ASC, date DESC"
        else:
            query += " ORDER BY date DESC"

        cur = conn.execute(query, params)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_email_by_id(message_id):
    conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM emails WHERE message_id = ?", (message_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_stats():
    """Summary statistics used by the Dashboard and Gmail Settings pages."""
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) AS c FROM emails").fetchone()["c"]
        by_priority = {
            row["priority"]: row["c"]
            for row in conn.execute(
                "SELECT priority, COUNT(*) AS c FROM emails GROUP BY priority"
            ).fetchall()
        }
        by_category = {
            row["category"]: row["c"]
            for row in conn.execute(
                "SELECT category, COUNT(*) AS c FROM emails GROUP BY category"
            ).fetchall()
        }
        last_synced_row = conn.execute(
            "SELECT MAX(last_synced) AS ts FROM emails"
        ).fetchone()
        last_synced = last_synced_row["ts"] if last_synced_row else None

        return {
            "total_emails": total,
            "high_priority": by_priority.get("HIGH", 0),
            "medium_priority": by_priority.get("MEDIUM", 0),
            "low_priority": by_priority.get("LOW", 0),
            "by_category": by_category,
            "last_synced": last_synced,
        }
    finally:
        conn.close()


def clear_all_data():
    """
    Delete ALL rows from InboxIQ's local cache table.
    This ONLY affects the local SQLite database - it never contacts Gmail
    and never deletes, modifies, or archives anything in the user's inbox.
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM emails")
        conn.commit()
    finally:
        conn.close()


# Ensure the table exists as soon as this module is imported anywhere.
init_db()
