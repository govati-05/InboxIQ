"""
InboxIQ - Gmail API Service (READ-ONLY)
==========================================
Handles OAuth 2.0 authentication and fetching emails from Gmail.

IMPORTANT: This module requests ONLY the 'gmail.readonly' scope and never
calls any Gmail API method that sends, deletes, modifies, archives, or
labels messages. It only reads.

Flow:
    1. Check for token.json (stored OAuth credentials).
    2. If missing/invalid/expired, run the OAuth consent flow using
       credentials.json (downloaded from Google Cloud Console).
    3. Save the resulting token to token.json for future runs.
    4. Build the Gmail API client and fetch messages.
"""

import base64
import os
from datetime import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.preprocessing import strip_html

# READ-ONLY scope only. InboxIQ never requests gmail.modify, gmail.send,
# or any scope that would allow changing the user's mailbox.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "credentials.json")
TOKEN_PATH = os.path.join(PROJECT_ROOT, "token.json")


class CredentialsNotFoundError(Exception):
    """Raised when credentials.json is missing from the project root."""
    pass


class GmailAuthError(Exception):
    """Raised when OAuth authentication fails for any reason."""
    pass


def get_gmail_service():
    """
    Returns an authenticated, read-only Gmail API service object.
    Raises CredentialsNotFoundError or GmailAuthError with a clear message
    on failure so the Streamlit UI can show a friendly error instead of a
    raw traceback.
    """
    creds = None

    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception:
            creds = None  # corrupted token.json -> re-authenticate below

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                creds = None
                if os.path.exists(TOKEN_PATH):
                    os.remove(TOKEN_PATH)
                raise GmailAuthError(
                    "Your saved Gmail session has expired and could not be refreshed "
                    "automatically. Please re-authorize InboxIQ."
                ) from exc
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise CredentialsNotFoundError(
                    "credentials.json was not found in the project root. "
                    "Download OAuth credentials from Google Cloud Console and place "
                    "the file at the root of the InboxIQ project (see README for steps)."
                )
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as exc:
                raise GmailAuthError(f"Gmail OAuth authorization failed: {exc}") from exc

        try:
            with open(TOKEN_PATH, "w", encoding="utf-8") as token_file:
                token_file.write(creds.to_json())
        except Exception:
            pass  # non-fatal: app still works this session even if we can't persist the token

    try:
        service = build("gmail", "v1", credentials=creds)
        return service
    except Exception as exc:
        raise GmailAuthError(f"Failed to connect to the Gmail API: {exc}") from exc


def is_authenticated():
    """Lightweight check used by the UI to show connection status without forcing a login."""
    if not os.path.exists(TOKEN_PATH):
        return False
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        return creds is not None and (creds.valid or (creds.expired and creds.refresh_token))
    except Exception:
        return False


def credentials_file_exists():
    return os.path.exists(CREDENTIALS_PATH)


def _get_header(headers, name):
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _decode_part_data(data):
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_body_from_payload(payload):
    """
    Recursively walk a Gmail message payload to extract readable text.
    Handles plain text, multipart, and HTML bodies (HTML is converted to
    plain readable text - no script execution, no attachment downloads).
    """
    if payload is None:
        return ""

    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})

    # Simple (non-multipart) message
    if mime_type == "text/plain" and body.get("data"):
        return _decode_part_data(body["data"])

    if mime_type == "text/html" and body.get("data"):
        html = _decode_part_data(body["data"])
        return strip_html(html)

    # Multipart message: prefer text/plain, fall back to text/html
    parts = payload.get("parts", [])
    if parts:
        plain_text = ""
        html_text = ""
        for part in parts:
            part_mime = part.get("mimeType", "")
            part_body = part.get("body", {})
            if part_mime == "text/plain" and part_body.get("data") and not plain_text:
                plain_text = _decode_part_data(part_body["data"])
            elif part_mime == "text/html" and part_body.get("data") and not html_text:
                html_text = strip_html(_decode_part_data(part_body["data"]))
            elif part_mime.startswith("multipart/"):
                nested = _extract_body_from_payload(part)
                if nested:
                    plain_text = plain_text or nested
        return plain_text or html_text

    return ""


def _parse_message(msg):
    """Convert a raw Gmail API message object into a flat dict InboxIQ can store/classify."""
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    sender = _get_header(headers, "From")
    recipient = _get_header(headers, "To")
    subject = _get_header(headers, "Subject") or "(No Subject)"
    date_header = _get_header(headers, "Date")

    body = _extract_body_from_payload(payload).strip()
    if not body:
        body = msg.get("snippet", "")

    # Normalize date: prefer Gmail's internalDate (epoch ms, always present & reliable)
    internal_date = msg.get("internalDate")
    if internal_date:
        try:
            date_iso = datetime.fromtimestamp(int(internal_date) / 1000).isoformat()
        except Exception:
            date_iso = date_header or ""
    else:
        date_iso = date_header or ""

    return {
        "message_id": msg.get("id"),
        "thread_id": msg.get("threadId"),
        "sender": sender,
        "recipient": recipient,
        "subject": subject,
        "body": body,
        "snippet": msg.get("snippet", ""),
        "date": date_iso,
        "labels": ",".join(msg.get("labelIds", []) or []),
    }


def fetch_emails(max_results=50, existing_ids=None):
    """
    Fetch up to `max_results` messages from the user's Gmail INBOX using
    read-only access. Messages whose id is already in `existing_ids`
    (already cached locally) are skipped to avoid re-downloading the full
    message body unnecessarily - only their id is counted.

    Returns a list of parsed email dicts (see _parse_message).
    Raises GmailAuthError / CredentialsNotFoundError on auth problems.
    """
    existing_ids = existing_ids or set()
    service = get_gmail_service()

    fetched = []
    try:
        next_page_token = None
        remaining = max_results

        while remaining > 0:
            page_size = min(remaining, 100)
            response = service.users().messages().list(
                userId="me",
                labelIds=["INBOX"],
                maxResults=page_size,
                pageToken=next_page_token,
            ).execute()

            message_stubs = response.get("messages", [])
            if not message_stubs:
                break

            for stub in message_stubs:
                msg_id = stub["id"]
                if msg_id in existing_ids:
                    continue
                try:
                    full_msg = service.users().messages().get(
                        userId="me", id=msg_id, format="full"
                    ).execute()
                    fetched.append(_parse_message(full_msg))
                except HttpError:
                    continue  # skip a single unreadable message rather than failing the whole sync

            remaining = max_results - len(fetched)
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return fetched

    except HttpError as exc:
        raise GmailAuthError(f"Gmail API request failed: {exc}") from exc
    except Exception as exc:
        raise GmailAuthError(f"Unexpected error while fetching Gmail messages: {exc}") from exc
