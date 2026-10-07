"""
InboxIQ - UI Components
==========================
Presentation helpers for the Streamlit front end: global CSS, page header,
KPI tiles, mail-client style email rows, and a shared Plotly chart style.

Everything here is purely visual - no ML, Gmail or database logic. All
user-supplied text (subjects, senders, bodies) is HTML-escaped before it is
rendered, so email content can never inject markup into the page.
"""

import hashlib
from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"

# Priority is an ordered status, so it uses the reserved status colours and is
# always shown together with its text label (never colour alone).
PRIORITY_COLOR = {"HIGH": "#d03b3b", "MEDIUM": "#fab219", "LOW": "#0ca30c"}
PRIORITY_LABEL = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}

# Fixed categorical order (CVD-validated) - used for avatars and multi-series charts.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "cross", "star", "hexagon", "pentagon", "triangle-down"]

CATEGORY_ICON = {
    "Education": "🎓", "Work": "💼", "Finance": "🏦", "Shopping": "🛍️", "Travel": "✈️",
    "Social": "💬", "Security": "🛡️", "Promotions": "🏷️", "Personal": "👤", "Events": "📅",
    "Newsletter": "📰", "Other": "📁",
}


CSS = """
<style>
:root {
  --iq-surface: #ffffff;
  --iq-plane: #f6f7f9;
  --iq-ink: #0b0b0b;
  --iq-ink-2: #52514e;
  --iq-muted: #898781;
  --iq-line: rgba(11, 11, 11, .09);
  --iq-accent: #2a78d6;
  --iq-accent-soft: rgba(42, 120, 214, .09);
  --iq-radius: 12px;
}
html, body, [class*="css"] { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Containers created with key="card_..." become white cards */
[class*="st-key-card_"] {
  background: var(--iq-surface); border: 1px solid var(--iq-line); border-radius: var(--iq-radius);
  padding: 1rem 1.1rem .6rem; box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
.iq-card-title { font-weight: 650; font-size: 1rem; color: var(--iq-ink); margin: 0 0 .1rem; }
.iq-card-sub { font-size: .8rem; color: var(--iq-muted); margin: 0 0 .5rem; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--iq-line); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
.iq-brand { display: flex; align-items: center; gap: .65rem; margin-bottom: .2rem; }
.iq-logo {
  width: 38px; height: 38px; border-radius: 10px; display: grid; place-items: center;
  background: linear-gradient(135deg, #2a78d6, #4a3aa7); color: #fff; font-weight: 800; font-size: 1.05rem;
}
.iq-brand-name { font-weight: 750; font-size: 1.2rem; color: var(--iq-ink); line-height: 1.1; }
.iq-brand-tag { font-size: .74rem; color: var(--iq-muted); }
.iq-side-label { font-size: .7rem; letter-spacing: .06em; text-transform: uppercase; color: var(--iq-muted); margin: 1rem 0 .3rem; font-weight: 600; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
  padding: .45rem .6rem; border-radius: 8px; margin: 0; width: 100%;
  transition: background .12s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: rgba(11, 11, 11, .04); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) { background: var(--iq-accent-soft); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p { color: var(--iq-accent); font-weight: 650; }
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none; }
.iq-account {
  display: flex; align-items: center; gap: .6rem; padding: .7rem; border-radius: 10px;
  background: var(--iq-plane); border: 1px solid var(--iq-line); margin-top: .4rem;
}
.iq-account .who { font-size: .85rem; font-weight: 600; color: var(--iq-ink); line-height: 1.2; }
.iq-account .mail { font-size: .74rem; color: var(--iq-muted); word-break: break-all; }
.iq-note { font-size: .76rem; color: var(--iq-ink-2); line-height: 1.45; margin-top: .8rem; }

/* Page header */
.iq-header { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.iq-title { font-size: 1.75rem; font-weight: 750; color: var(--iq-ink); margin: 0; line-height: 1.2; }
.iq-subtitle { color: var(--iq-ink-2); font-size: .95rem; margin-top: .25rem; }
.iq-pill {
  display: inline-flex; align-items: center; gap: .45rem; padding: .35rem .75rem; border-radius: 999px;
  font-size: .8rem; font-weight: 600; white-space: nowrap;
}
.iq-pill.demo { background: var(--iq-accent-soft); color: #1c5cab; border: 1px solid rgba(42, 120, 214, .25); }
.iq-pill.live { background: rgba(12, 163, 12, .08); color: #006300; border: 1px solid rgba(12, 163, 12, .25); }
.iq-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

/* KPI tiles */
.iq-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem; margin-bottom: 1.2rem; }
.iq-kpi {
  background: var(--iq-surface); border: 1px solid var(--iq-line); border-radius: var(--iq-radius);
  padding: .9rem 1rem; box-shadow: 0 1px 2px rgba(16, 24, 40, .04); position: relative; overflow: hidden;
}
.iq-kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--kpi, var(--iq-accent)); }
.iq-kpi .label { font-size: .78rem; color: var(--iq-ink-2); display: flex; align-items: center; gap: .35rem; font-weight: 550; }
.iq-kpi .value { font-size: 1.85rem; font-weight: 700; color: var(--iq-ink); line-height: 1.15; margin-top: .25rem; }
.iq-kpi .hint { font-size: .74rem; color: var(--iq-muted); margin-top: .1rem; }

/* Email rows */
.iq-mail {
  display: grid; grid-template-columns: 40px 1fr auto; gap: .8rem; align-items: start;
  background: var(--iq-surface); border: 1px solid var(--iq-line); border-radius: 10px;
  padding: .75rem .9rem; margin-bottom: .1rem; position: relative;
}
.iq-mail.unread { box-shadow: inset 3px 0 0 var(--iq-accent); }
.iq-avatar {
  width: 38px; height: 38px; border-radius: 50%; display: grid; place-items: center;
  color: #fff; font-weight: 700; font-size: .85rem; letter-spacing: .02em;
}
.iq-mail .top { display: flex; align-items: baseline; gap: .5rem; min-width: 0; }
.iq-mail .sender { font-size: .9rem; color: var(--iq-ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.iq-mail.unread .sender, .iq-mail.unread .subject { font-weight: 700; }
.iq-mail .addr { font-size: .76rem; color: var(--iq-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.iq-mail .subject { font-size: .93rem; color: var(--iq-ink); margin-top: .1rem; font-weight: 550; }
.iq-mail .snippet { font-size: .84rem; color: var(--iq-ink-2); margin-top: .15rem; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.iq-mail .chips { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .45rem; }
.iq-mail .meta { text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: .35rem; }
.iq-mail .time { font-size: .78rem; color: var(--iq-ink-2); white-space: nowrap; }
.iq-mail.unread .time { color: var(--iq-accent); font-weight: 650; }
.iq-star { color: #eda100; font-size: .95rem; line-height: 1; }
.iq-chip {
  display: inline-flex; align-items: center; gap: .3rem; font-size: .72rem; font-weight: 600;
  padding: .15rem .5rem; border-radius: 6px; background: #f1f1ee; color: var(--iq-ink-2); white-space: nowrap;
}
.iq-chip .dot { width: 7px; height: 7px; border-radius: 50%; }
.iq-score { display: inline-flex; align-items: center; gap: .4rem; font-size: .72rem; color: var(--iq-ink-2); font-weight: 600; }
.iq-score .track { width: 56px; height: 6px; border-radius: 4px; background: #ecebe6; overflow: hidden; }
.iq-score .fill { height: 100%; border-radius: 4px; }
@media (max-width: 640px) {
  .iq-mail { grid-template-columns: 34px 1fr; }
  .iq-mail .meta { grid-column: 2; flex-direction: row; align-items: center; justify-content: flex-start; }
  .iq-avatar { width: 32px; height: 32px; font-size: .75rem; }
}

/* Detail panel inside an opened email */
.iq-detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .6rem; margin: .3rem 0 .8rem; }
.iq-detail { background: var(--iq-plane); border-radius: 8px; padding: .55rem .7rem; }
.iq-detail .k { font-size: .7rem; color: var(--iq-muted); text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
.iq-detail .v { font-size: .92rem; color: var(--iq-ink); font-weight: 600; margin-top: .1rem; }
.iq-body { white-space: pre-wrap; font-size: .9rem; color: var(--iq-ink); line-height: 1.55; background: var(--iq-surface); border: 1px solid var(--iq-line); border-radius: 8px; padding: .8rem .9rem; }

/* Score gauge used on Analyze Email */
.iq-gauge { height: 12px; border-radius: 6px; background: linear-gradient(90deg, #0ca30c 0%, #fab219 50%, #d03b3b 100%); position: relative; margin: .8rem 0 .3rem; }
.iq-gauge .needle { position: absolute; top: -5px; width: 4px; height: 22px; border-radius: 2px; background: var(--iq-ink); transform: translateX(-2px); box-shadow: 0 0 0 2px #fff; }
.iq-gauge-scale { display: flex; justify-content: space-between; font-size: .72rem; color: var(--iq-muted); }

.iq-empty { text-align: center; padding: 1.6rem 1rem; color: var(--iq-ink-2); font-size: .9rem; }
[data-testid="stExpander"] details { border-radius: 10px; border-color: var(--iq-line); }
/* Each email = row + attached "open" drawer */
[class*="st-key-mail_"] { gap: 0 !important; margin-bottom: .55rem; }
[class*="st-key-mail_"] .iq-mail { border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0; }
[class*="st-key-mail_"] [data-testid="stExpander"] details {
  border-top: 0; border-top-left-radius: 0; border-top-right-radius: 0; background: #fbfbfa;
}
[class*="st-key-mail_"] [data-testid="stExpander"] summary { padding-top: .35rem; padding-bottom: .35rem; }
[class*="st-key-mail_"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
[class*="st-key-mail_"] .stMarkdown { margin-bottom: 0; }
[data-testid="stExpander"] summary p { font-size: .82rem; color: var(--iq-ink-2); }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def friendly_date(value):
    """'09:42' for today, 'Yesterday', weekday for this week, else '05 Oct' - like a mail client."""
    if value is None or pd.isna(value):
        return ""
    ts = pd.Timestamp(value)
    today = datetime.now().date()
    days = (today - ts.date()).days
    if days == 0:
        return ts.strftime("%H:%M")
    if days == 1:
        return "Yesterday"
    if 1 < days < 7:
        return ts.strftime("%a")
    return ts.strftime("%d %b")


def initials(name):
    parts = [p for p in str(name).replace(".", " ").split() if p[:1].isalpha()]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def avatar_color(key):
    digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    return SERIES[int(digest, 16) % len(SERIES)]


def _num(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def priority_chip(priority):
    color = PRIORITY_COLOR.get(priority, AXIS)
    label = PRIORITY_LABEL.get(priority, priority or "Unknown")
    return f'<span class="iq-chip"><span class="dot" style="background:{color}"></span>{escape(label)} priority</span>'


def category_chip(category):
    icon = CATEGORY_ICON.get(category, "📁")
    return f'<span class="iq-chip">{icon} {escape(str(category or "Other"))}</span>'


def score_bar(score, priority):
    s = _num(score)
    if s is None:
        return ""
    color = PRIORITY_COLOR.get(priority, ACCENT)
    return (f'<span class="iq-score" title="Priority score {s:.0f}/100"><span class="track">'
            f'<span class="fill" style="width:{max(min(s, 100), 0):.0f}%;background:{color}"></span></span>{s:.0f}</span>')


# ---------------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------------

def page_header(title, subtitle="", demo=True):
    pill = ('<span class="iq-pill demo"><span class="dot"></span>Demo Mode — Showing Sample Emails</span>' if demo
            else '<span class="iq-pill live"><span class="dot"></span>Gmail · read-only</span>')
    sub = f'<div class="iq-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="iq-header"><div><h1 class="iq-title">{escape(title)}</h1>{sub}</div>{pill}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items: list of (label, value, hint, color, icon)."""
    tiles = ""
    for label, value, hint, color, icon in items:
        value = str(value)
        size = ' style="font-size:1.3rem;padding-top:.35rem"' if len(value) > 10 else ""  # long text values
        tiles += (
            f'<div class="iq-kpi" style="--kpi:{color}"><div class="label">{icon} {escape(label)}</div>'
            f'<div class="value"{size}>{escape(value)}</div><div class="hint">{escape(hint)}</div></div>'
        )
    st.markdown(f'<div class="iq-kpis">{tiles}</div>', unsafe_allow_html=True)


def card_title(title, subtitle=""):
    sub = f'<div class="iq-card-sub">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="iq-card-title">{escape(title)}</div>{sub}', unsafe_allow_html=True)


def empty_state(text):
    st.markdown(f'<div class="iq-empty">{escape(text)}</div>', unsafe_allow_html=True)


def email_row_html(email):
    unread = not bool(email.get("is_read", True))
    starred = bool(email.get("is_starred", False))
    name = email.get("sender_name") or email.get("sender") or "Unknown"
    addr = email.get("sender_email") or ""
    snippet = email.get("snippet") or str(email.get("body", ""))[:200]
    star = '<span class="iq-star" title="Important">★</span>' if starred else ""
    addr_html = f'<span class="addr">{escape(addr)}</span>' if addr and addr != name else ""
    return (
        f'<div class="iq-mail{" unread" if unread else ""}">'
        f'<div class="iq-avatar" style="background:{avatar_color(addr or name)}">{escape(initials(name))}</div>'
        f'<div style="min-width:0"><div class="top"><span class="sender">{escape(name)}</span>{addr_html}</div>'
        f'<div class="subject">{escape(str(email.get("subject") or "(No Subject)"))}</div>'
        f'<div class="snippet">{escape(str(snippet))}</div>'
        f'<div class="chips">{category_chip(email.get("category"))}{priority_chip(email.get("priority"))}</div></div>'
        f'<div class="meta"><span class="time">{escape(friendly_date(email.get("date_parsed")))}</span>'
        f'{star}{score_bar(email.get("priority_score"), email.get("priority"))}</div>'
        f'</div>'
    )


def email_detail(email, key_prefix=""):
    """Expandable 'open email' panel: full body plus the ML prediction details."""
    conf = _num(email.get("confidence"))
    score = _num(email.get("priority_score"))
    when = email.get("date_parsed")
    when_str = pd.Timestamp(when).strftime("%a, %d %b %Y · %H:%M") if when is not None and not pd.isna(when) else str(email.get("date", ""))
    with st.expander("Open email & AI insights"):
        cells = [
            ("Category", f'{CATEGORY_ICON.get(email.get("category"), "📁")} {email.get("category") or "—"}'),
            ("Confidence", f"{conf * 100:.0f}%" if conf is not None else "—"),
            ("Priority", PRIORITY_LABEL.get(email.get("priority"), "—")),
            ("Priority score", f"{score:.0f} / 100" if score is not None else "—"),
            ("Predicted by", email.get("prediction_source") or "ML model"),
        ]
        if email.get("expected_category"):
            cells.append(("Reference label", f'{email["expected_category"]} · {PRIORITY_LABEL.get(email["expected_priority"], "")}'))
        grid = "".join(f'<div class="iq-detail"><div class="k">{escape(k)}</div><div class="v">{escape(str(v))}</div></div>'
                       for k, v in cells)
        st.markdown(f'<div class="iq-detail-grid">{grid}</div>', unsafe_allow_html=True)
        st.caption(f'From {email.get("sender", "Unknown")} · To {email.get("recipient") or "me"} · {when_str}')
        st.markdown(f'<div class="iq-body">{escape(str(email.get("body") or ""))}</div>', unsafe_allow_html=True)
        st.caption(f'Message ID {email.get("message_id", "—")} · Thread {email.get("thread_id", "—")}')


def email_list(df, key_prefix, with_details=True, limit=None):
    rows = df if limit is None else df.head(limit)
    if rows.empty:
        empty_state("No emails here.")
        return
    for _, email in rows.iterrows():
        with st.container(key=f"mail_{key_prefix}_{email.get('message_id')}"):
            st.markdown(email_row_html(email), unsafe_allow_html=True)
            if with_details:
                email_detail(email, key_prefix=key_prefix)


def score_gauge(score):
    st.markdown(
        f'<div class="iq-gauge"><div class="needle" style="left:{max(min(score, 100), 0)}%"></div></div>'
        f'<div class="iq-gauge-scale"><span>0 · Low</span><span>50</span><span>High · 100</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def style_fig(fig, height=300, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=4, r=12, t=8 if not legend else 30, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=12),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, color=INK, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None, font=dict(size=12)),
        bargap=0.28, barcornerradius=4,
    )
    fig.update_xaxes(automargin=True, gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(automargin=True, gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED, size=12))
    return fig


def show_chart(fig, key=None):
    st.plotly_chart(fig, key=key, theme=None, config={"displayModeBar": False, "responsive": True})
