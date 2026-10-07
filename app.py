"""
InboxIQ - Intelligent Gmail Email Classification & Priority Management System
================================================================================
Streamlit application entry point.

Run:
    streamlit run app.py

Demo Mode (default ON) shows a built-in, fully fictional sample inbox
(src/demo_data.py) classified live by the trained ML models - no Gmail
account, OAuth or internet connection needed. Turn Demo Mode OFF in the
sidebar to use the real read-only Gmail integration.

Visual components (CSS, email rows, KPI tiles, chart styling) live in
src/ui.py; this file holds the page logic.
"""

import os
import sys
from datetime import datetime
from email.utils import parseaddr
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src import database as db
from src import ui
from src.predict import classify_email, models_available, get_metrics, ModelsNotTrainedError
from src.gmail_service import (
    fetch_emails, is_authenticated, credentials_file_exists,
    CredentialsNotFoundError, GmailAuthError,
)
from src.clustering import run_clustering
from src.demo_data import get_demo_emails, DEMO_ACCOUNT_NAME, DEMO_ACCOUNT_EMAIL
from src.evaluation import evaluate_predictions

st.set_page_config(page_title="InboxIQ", page_icon="📬", layout="wide")
ui.inject_css()

CATEGORIES = [
    "Education", "Work", "Finance", "Shopping", "Travel", "Social",
    "Security", "Promotions", "Personal", "Events", "Newsletter", "Other",
]
PRIORITIES = ["HIGH", "MEDIUM", "LOW"]

# Sequential blue ramp (light -> dark) for heatmaps.
BLUES = [[0, "#f3f7fd"], [0.15, "#cde2fb"], [0.4, "#86b6ef"], [0.7, "#2a78d6"], [1, "#104281"]]

# Demo Mode is ON by default so the app works immediately after install.
# Set the environment variable INBOXIQ_DEMO_MODE=0 to start with it OFF.
DEMO_MODE_DEFAULT = os.environ.get("INBOXIQ_DEMO_MODE", "1").strip() not in ("0", "false", "off", "no")


# -------------------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------------------

@st.cache_data(show_spinner="Classifying your inbox with InboxIQ's ML models...", ttl=3600)
def load_demo_emails(use_models=True):
    """
    Load the built-in sample inbox for Demo Mode (no Gmail needed) and run
    every email through the real ML pipeline. Results stay in memory only and
    are never written to the SQLite Gmail cache.

    If the models are not trained (or cannot be loaded), the hand-assigned
    reference labels are used instead so the dashboard still works.
    """
    rows = get_demo_emails()
    df = pd.DataFrame(rows)
    if use_models:
        try:
            return classify_and_cache_demo(df)
        except Exception:
            pass  # fall through to reference labels
    df["category"] = df["expected_category"]
    df["priority"] = df["expected_priority"]
    df["priority_score"] = df["expected_score"]
    df["confidence"] = None
    df["prediction_source"] = "Reference label"
    return df


def classify_and_cache_demo(df):
    """Run the real ML pipeline over demo emails (in-memory only, not written to SQLite)."""
    results = []
    for _, row in df.iterrows():
        result = classify_email(row["subject"], row["body"])
        results.append({**row.to_dict(), **{
            "category": result["category"],
            "priority": result["priority"],
            "priority_score": result["priority_score"],
            "confidence": result["category_confidence"],
            "prediction_source": "ML model",
        }})
    return pd.DataFrame(results)


def sync_gmail(max_results):
    """Fetch new Gmail messages, classify them with the trained ML models, and cache them in SQLite."""
    existing_ids = db.get_existing_message_ids()
    emails = fetch_emails(max_results=max_results, existing_ids=existing_ids)

    newly_added = 0
    for email in emails:
        try:
            result = classify_email(email["subject"], email["body"])
        except ModelsNotTrainedError:
            raise
        record = {
            **email,
            "category": result["category"],
            "priority": result["priority"],
            "priority_score": result["priority_score"],
            "confidence": result["category_confidence"],
        }
        db.upsert_email(record)
        newly_added += 1
    return newly_added, len(existing_ids)


def prepare_dataframe(df):
    """
    Add the display columns the UI needs (sender name/address, read/starred
    flags, parsed dates) so Demo Mode and Gmail Mode render the same way.
    Gmail rows derive read/starred status from their Gmail labels.
    """
    if df.empty:
        return df
    df = df.copy()
    labels = df["labels"].fillna("").astype(str) if "labels" in df else pd.Series("", index=df.index)

    if "sender_name" not in df or "sender_email" not in df:
        parsed = df["sender"].fillna("").astype(str).apply(parseaddr)
        df["sender_email"] = [addr or raw for (name, addr), raw in zip(parsed, df["sender"].fillna(""))]
        df["sender_name"] = [name or addr or "Unknown" for name, addr in parsed]
    if "is_read" not in df:
        df["is_read"] = ~labels.str.contains("UNREAD")
    if "is_starred" not in df:
        df["is_starred"] = labels.str.contains("STARRED") | labels.str.contains("IMPORTANT")
    if "prediction_source" not in df:
        df["prediction_source"] = "ML model"

    df["is_read"] = df["is_read"].fillna(True).astype(bool)
    df["is_starred"] = df["is_starred"].fillna(False).astype(bool)
    df["priority_score"] = pd.to_numeric(df["priority_score"], errors="coerce")
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed")
    df["date_parsed"] = dates.dt.tz_convert(None)
    return df


def by_priority(df):
    """Sort HIGH -> LOW, then by score, then newest first."""
    return df.assign(_rank=df["priority"].map({"HIGH": 0, "MEDIUM": 1, "LOW": 2}).fillna(3)).sort_values(
        ["_rank", "priority_score", "date_parsed"], ascending=[True, False, False])


def card(name):
    """A white card container (styled in src/ui.py via its key)."""
    return st.container(key=f"card_{name}")


def greeting():
    hour = datetime.now().hour
    part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
    return f"Good {part}, {DEMO_ACCOUNT_NAME.split()[0]}" if mode == "Demo Mode" else f"Good {part}"


# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="iq-brand"><div class="iq-logo">IQ</div><div>'
        '<div class="iq-brand-name">InboxIQ</div>'
        '<div class="iq-brand-tag">Smart email classification & priority</div></div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    demo_on = st.toggle(
        "Demo Mode", value=DEMO_MODE_DEFAULT, key="demo_mode",
        help="ON: use the built-in sample inbox - no Gmail login or internet needed. "
             "OFF: use your real Gmail inbox (read-only) via Gmail Settings.",
    )
    st.caption("Demo Mode: **ON** — sample inbox, works offline" if demo_on
               else "Demo Mode: **OFF** — using your Gmail inbox")
    mode = "Demo Mode" if demo_on else "Gmail Mode"

    st.markdown('<div class="iq-side-label">Menu</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📥 Smart Inbox", "🤖 Analyze Email", "📊 ML Analytics",
         "🔬 Email Clusters", "⚙️ Gmail Settings", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="iq-side-label">Account</div>', unsafe_allow_html=True)
    if mode == "Demo Mode":
        who, mail, color = DEMO_ACCOUNT_NAME, DEMO_ACCOUNT_EMAIL, ui.avatar_color(DEMO_ACCOUNT_EMAIL)
        avatar = ui.initials(DEMO_ACCOUNT_NAME)
        note = ("Demo Mode uses a built-in, fully fictional sample inbox. No Gmail account is "
                "accessed and everything runs offline on this machine.")
    else:
        connected = is_authenticated()
        who, mail, color, avatar = "Gmail", ("Connected · read-only" if connected else "Not connected"), "#52514e", "G"
        note = ("Gmail is accessed only after Google OAuth authorization, with read-only scope. "
                "InboxIQ never sends, deletes, modifies, archives or labels emails. Data is "
                "processed and cached locally in SQLite.")
    st.markdown(
        f'<div class="iq-account"><div class="iq-avatar" style="background:{color}">{escape(avatar)}</div>'
        f'<div><div class="who">{escape(who)}</div><div class="mail">{escape(mail)}</div></div></div>'
        f'<div class="iq-note">🔒 {escape(note)}</div>',
        unsafe_allow_html=True,
    )

    if not models_available():
        st.error("ML models are not trained yet. Run `python src/train_models.py` or:")
        if st.button("Train models now", width="stretch"):
            from src import train_models
            with st.spinner("Training models on the synthetic dataset..."):
                train_models.main()
            st.cache_data.clear()
            st.rerun()


# -------------------------------------------------------------------------
# Data loading based on mode
# -------------------------------------------------------------------------

def get_active_dataframe():
    if mode == "Demo Mode":
        df = load_demo_emails(use_models=models_available())
    else:
        df = pd.DataFrame(db.get_all_emails())
    return prepare_dataframe(df)


def models_required():
    """Gmail Mode needs trained models; Demo Mode can fall back to reference labels."""
    return mode == "Gmail Mode" and not models_available()


def no_emails_warning():
    st.info(
        "Your inbox is empty. "
        + ("Open **⚙️ Gmail Settings** and click **Sync Gmail**, or turn **Demo Mode** ON in the sidebar."
           if mode == "Gmail Mode" else "The demo inbox could not be loaded.")
    )


IS_DEMO = mode == "Demo Mode"


# -------------------------------------------------------------------------
# Charts
# -------------------------------------------------------------------------

def chart_category_counts(df):
    counts = df["category"].value_counts().sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=[f"{ui.CATEGORY_ICON.get(c, '')} {c}" for c in counts.index], orientation="h",
        marker_color=ui.ACCENT, text=counts.values, textposition="outside", cliponaxis=False,
        textfont=dict(color=ui.INK_2), hovertemplate="%{y}: %{x} emails<extra></extra>",
    ))
    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False)
    return ui.style_fig(fig, height=max(240, 28 * len(counts)), legend=False)


def chart_priority_donut(df):
    counts = df["priority"].value_counts().reindex(PRIORITIES).fillna(0)
    fig = go.Figure(go.Pie(
        labels=[ui.PRIORITY_LABEL[p] for p in PRIORITIES], values=counts.values, hole=.64, sort=False,
        marker=dict(colors=[ui.PRIORITY_COLOR[p] for p in PRIORITIES], line=dict(color="#ffffff", width=2)),
        textinfo="percent", textfont=dict(color="#ffffff", size=12),
        hovertemplate="%{label} priority: %{value} emails (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b style='font-size:22px;color:{ui.INK}'>{int(counts.sum())}</b><br>emails",
                       showarrow=False, font=dict(color=ui.MUTED, size=12))
    return ui.style_fig(fig, height=260)


def chart_category_priority(df):
    pivot = pd.crosstab(df["category"], df["priority"]).reindex(columns=PRIORITIES, fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for p in reversed(PRIORITIES):
        fig.add_bar(y=pivot.index, x=pivot[p], orientation="h", name=ui.PRIORITY_LABEL[p],
                    marker=dict(color=ui.PRIORITY_COLOR[p], line=dict(color="#ffffff", width=2)),
                    hovertemplate="%{y} · " + ui.PRIORITY_LABEL[p] + ": %{x}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Emails", dtick=1 if pivot.values.max() < 8 else None)
    fig.update_yaxes(showgrid=False)
    return ui.style_fig(fig, height=max(260, 28 * len(pivot)))


def chart_score_histogram(df):
    fig = go.Figure()
    for p in reversed(PRIORITIES):
        vals = df.loc[df["priority"] == p, "priority_score"].dropna()
        fig.add_histogram(x=vals, name=ui.PRIORITY_LABEL[p], xbins=dict(start=0, end=100, size=10),
                          marker=dict(color=ui.PRIORITY_COLOR[p], line=dict(color="#ffffff", width=2)),
                          hovertemplate="Score %{x}: %{y} emails<extra>" + ui.PRIORITY_LABEL[p] + "</extra>")
    fig.update_layout(barmode="stack", bargap=0.08)
    fig.update_xaxes(title="Priority score", range=[0, 100], dtick=10, showgrid=False)
    fig.update_yaxes(title="Emails")
    return ui.style_fig(fig, height=300)


def chart_avg_score_by_category(df):
    avg = df.groupby("category")["priority_score"].mean().sort_values()
    fig = go.Figure(go.Bar(
        x=avg.values, y=[f"{ui.CATEGORY_ICON.get(c, '')} {c}" for c in avg.index], orientation="h",
        marker_color=ui.ACCENT, text=[f"{v:.0f}" for v in avg.values], textposition="outside", cliponaxis=False,
        textfont=dict(color=ui.INK_2), hovertemplate="%{y}: average score %{x:.1f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 108], showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False)
    return ui.style_fig(fig, height=max(260, 28 * len(avg)), legend=False)


def chart_activity(df):
    d = df.dropna(subset=["date_parsed"])
    daily = d.groupby(d["date_parsed"].dt.normalize()).size()
    if not daily.empty:
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0)
    fig = go.Figure(go.Scatter(
        x=daily.index, y=daily.values, mode="lines+markers", line=dict(color=ui.ACCENT, width=2),
        marker=dict(size=8, color=ui.ACCENT, line=dict(color="#ffffff", width=2)),
        fill="tozeroy", fillcolor="rgba(42,120,214,.10)", name="Emails",
        hovertemplate="%{x|%a %d %b}: %{y} emails<extra></extra>",
    ))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showgrid=False, tickformat="%d %b")
    fig.update_yaxes(title="Emails received", rangemode="tozero")
    return ui.style_fig(fig, height=300, legend=False)


def chart_read_unread(df):
    pivot = pd.crosstab(df["category"], df["is_read"].map({True: "Read", False: "Unread"}))
    pivot = pivot.reindex(columns=["Unread", "Read"], fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for col, color in [("Unread", ui.ACCENT), ("Read", "#cfd3da")]:
        fig.add_bar(y=pivot.index, x=pivot[col], orientation="h", name=col,
                    marker=dict(color=color, line=dict(color="#ffffff", width=2)),
                    hovertemplate="%{y} · " + col + ": %{x}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Emails")
    fig.update_yaxes(showgrid=False)
    return ui.style_fig(fig, height=max(260, 28 * len(pivot)))


def chart_confusion(cm, labels, y_title="Actual"):
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, colorscale=BLUES, showscale=False, xgap=2, ygap=2,
        text=[[str(v) if v else "" for v in row] for row in cm], texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate=y_title + " %{y} → predicted %{x}: %{z}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title=y_title, showgrid=False)
    fig.update_xaxes(title="Predicted", showgrid=False, tickangle=-35)
    return ui.style_fig(fig, height=max(320, 34 * len(labels)), legend=False)


def chart_model_comparison(task):
    models = list(task["results"].keys())
    fig = go.Figure()
    for i, (key, label) in enumerate([("accuracy", "Accuracy"), ("precision", "Precision"),
                                      ("recall", "Recall"), ("f1_score", "F1-score")]):
        fig.add_bar(x=models, y=[task["results"][m][key] for m in models], name=label,
                    marker=dict(color=ui.SERIES[i], line=dict(color="#ffffff", width=2)),
                    hovertemplate="%{x} · " + label + ": %{y:.4f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_xaxes(showgrid=False)
    return ui.style_fig(fig, height=320)


# -------------------------------------------------------------------------
# PAGE: Dashboard
# -------------------------------------------------------------------------

if page == "🏠 Dashboard":
    if models_required():
        ui.page_header("Dashboard", demo=IS_DEMO)
        st.error("ML models are not trained yet. Please run `python src/train_models.py` first.")
    else:
        df = get_active_dataframe()
        if df.empty:
            ui.page_header("Dashboard", demo=IS_DEMO)
            no_emails_warning()
        else:
            total = len(df)
            unread = int((~df["is_read"]).sum())
            high = int((df["priority"] == "HIGH").sum())
            starred = int(df["is_starred"].sum())
            attention = by_priority(df[(df["priority"] == "HIGH") & (~df["is_read"])])
            scores = df["priority_score"].dropna()

            ui.page_header(
                greeting(),
                f"{datetime.now():%A, %d %B} · {unread} unread · {len(attention)} need your attention",
                demo=IS_DEMO,
            )
            if IS_DEMO and (df["prediction_source"] == "Reference label").all():
                st.caption("ML models not trained - showing reference labels. Train the models to see live predictions.")

            ui.kpi_row([
                ("Total emails", total, f"{df['category'].nunique()} categories", "#52514e", "📨"),
                ("Unread", unread, f"{unread / total * 100:.0f}% of inbox", ui.ACCENT, "🔵"),
                ("High priority", high, f"{len(attention)} still unread", ui.PRIORITY_COLOR["HIGH"], "🔴"),
                ("Medium priority", int((df["priority"] == "MEDIUM").sum()), "plan for today",
                 ui.PRIORITY_COLOR["MEDIUM"], "🟠"),
                ("Low priority", int((df["priority"] == "LOW").sum()), "read when free",
                 ui.PRIORITY_COLOR["LOW"], "🟢"),
                ("Important", starred, "starred emails", "#eda100", "⭐"),
            ])

            left, right = st.columns([3, 2], gap="medium")
            with left:
                with card("attention"):
                    ui.card_title("🚨 Needs your attention", "Unread, high-priority emails ranked by priority score")
                    if attention.empty:
                        ui.empty_state("You're all caught up — no unread high-priority emails.")
                    else:
                        ui.email_list(attention, key_prefix="attention", limit=5)
            with right:
                with card("mix"):
                    ui.card_title("Priority mix", "Share of emails at each predicted priority")
                    ui.show_chart(chart_priority_donut(df), key="donut")
                with card("score_summary"):
                    ui.card_title("Priority score", "0 = can wait · 100 = act now")
                    if scores.empty:
                        ui.empty_state("No priority scores available.")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Average", f"{scores.mean():.1f}")
                        c2.metric("Median", f"{scores.median():.0f}")
                        c3.metric("Range", f"{scores.min():.0f}–{scores.max():.0f}")

            st.write("")
            tab1, tab2, tab3 = st.tabs(["📂 Categories", "🎯 Priority scores", "📈 Activity"])
            with tab1:
                c1, c2 = st.columns(2, gap="medium")
                with c1, card("cat_counts"):
                    ui.card_title("Emails by category", "Predicted by the category model")
                    ui.show_chart(chart_category_counts(df), key="cat_counts")
                with c2, card("cat_pri"):
                    ui.card_title("Category vs priority", "How urgent each category tends to be")
                    ui.show_chart(chart_category_priority(df), key="cat_pri")
            with tab2:
                if scores.empty:
                    ui.empty_state("No priority scores available.")
                else:
                    ui.kpi_row([
                        ("Average score", f"{scores.mean():.1f}", "across all emails", ui.ACCENT, "∅"),
                        ("Median score", f"{scores.median():.0f}", "middle of the inbox", ui.ACCENT, "≈"),
                        ("Highest", f"{scores.max():.0f}", "most urgent email", ui.PRIORITY_COLOR["HIGH"], "▲"),
                        ("Lowest", f"{scores.min():.0f}", "least urgent email", ui.PRIORITY_COLOR["LOW"], "▼"),
                        ("Std. deviation", f"{scores.std():.1f}" if len(scores) > 1 else "0", "spread of scores",
                         "#52514e", "σ"),
                    ])
                    c1, c2 = st.columns(2, gap="medium")
                    with c1, card("hist"):
                        ui.card_title("Score distribution", "Number of emails per 10-point score band")
                        ui.show_chart(chart_score_histogram(df), key="hist")
                    with c2, card("avg_cat"):
                        ui.card_title("Average score by category", "Which categories carry the most urgency")
                        ui.show_chart(chart_avg_score_by_category(df), key="avg_cat")
            with tab3:
                c1, c2 = st.columns(2, gap="medium")
                with c1, card("activity"):
                    ui.card_title("Emails over time", "Emails received per day")
                    ui.show_chart(chart_activity(df), key="activity")
                with c2, card("read_unread"):
                    ui.card_title("Read vs unread", "Unread emails by category")
                    ui.show_chart(chart_read_unread(df), key="read_unread")

            st.write("")
            c1, c2 = st.columns(2, gap="medium")
            with c1, card("recent"):
                ui.card_title("📨 Recent emails", "Latest messages in your inbox")
                ui.email_list(df.sort_values("date_parsed", ascending=False), key_prefix="recent", limit=6)
            with c2, card("important"):
                ui.card_title("⭐ Important", "Starred emails")
                important = df[df["is_starred"]].sort_values("date_parsed", ascending=False)
                if important.empty:
                    ui.empty_state("No emails are marked important.")
                else:
                    ui.email_list(important, key_prefix="important", limit=6)


# -------------------------------------------------------------------------
# PAGE: Smart Inbox
# -------------------------------------------------------------------------

elif page == "📥 Smart Inbox":
    if models_required():
        ui.page_header("Smart Inbox", demo=IS_DEMO)
        st.error("ML models are not trained yet. Please run `python src/train_models.py` first.")
    else:
        df = get_active_dataframe()
        if df.empty:
            ui.page_header("Smart Inbox", demo=IS_DEMO)
            no_emails_warning()
        else:
            ui.page_header("Smart Inbox",
                           f"{len(df)} emails · {int((~df['is_read']).sum())} unread · sorted by what matters most",
                           demo=IS_DEMO)

            views = {
                "All": df,
                "Unread": df[~df["is_read"]],
                "Important": df[df["is_starred"]],
                "High priority": df[df["priority"] == "HIGH"],
            }
            view = st.segmented_control(
                "View", list(views.keys()), default="All", key="inbox_view", label_visibility="collapsed",
                format_func=lambda v: f"{v}  {len(views[v])}" if v in views else str(v),
            ) or "All"

            col1, col2, col3, col4 = st.columns([3, 1.3, 1.3, 1.3])
            with col1:
                search_term = st.text_input("Search", placeholder="🔎  Search subject, sender or text",
                                            label_visibility="collapsed")
            with col2:
                category_filter = st.selectbox("Category", ["All categories"] + CATEGORIES, label_visibility="collapsed")
            with col3:
                priority_filter = st.selectbox("Priority", ["All priorities"] + PRIORITIES,
                                               format_func=lambda p: ui.PRIORITY_LABEL.get(p, p),
                                               label_visibility="collapsed")
            with col4:
                sort_by = st.selectbox("Sort by", ["Priority", "Newest", "Oldest", "Category"],
                                       format_func=lambda s: f"Sort: {s}", label_visibility="collapsed")

            filtered = views[view].copy()
            if search_term:
                mask = (
                    filtered["subject"].astype(str).str.contains(search_term, case=False, na=False, regex=False)
                    | filtered["sender"].astype(str).str.contains(search_term, case=False, na=False, regex=False)
                    | filtered["body"].astype(str).str.contains(search_term, case=False, na=False, regex=False)
                )
                filtered = filtered[mask]
            if category_filter in CATEGORIES:
                filtered = filtered[filtered["category"] == category_filter]
            if priority_filter in PRIORITIES:
                filtered = filtered[filtered["priority"] == priority_filter]

            sort_map = {
                "Priority": by_priority,
                "Newest": lambda d: d.sort_values("date_parsed", ascending=False),
                "Oldest": lambda d: d.sort_values("date_parsed", ascending=True),
                "Category": lambda d: d.sort_values(["category", "date_parsed"], ascending=[True, False]),
            }
            filtered = sort_map[sort_by](filtered)

            page_size = 25
            shown = st.session_state.get("inbox_shown", page_size)
            st.caption(f"Showing {min(shown, len(filtered))} of {len(filtered)} emails")
            if filtered.empty:
                ui.empty_state("No emails match these filters.")
            else:
                ui.email_list(filtered, key_prefix="inbox", limit=shown)
                if len(filtered) > shown:
                    if st.button(f"Show more ({len(filtered) - shown} remaining)"):
                        st.session_state["inbox_shown"] = shown + page_size
                        st.rerun()


# -------------------------------------------------------------------------
# PAGE: Analyze Email
# -------------------------------------------------------------------------

elif page == "🤖 Analyze Email":
    ui.page_header("Analyze Email", "Paste any email to see how InboxIQ's ML models classify and score it",
                   demo=IS_DEMO)

    if not models_available():
        st.error("ML models are not trained yet. Please run `python src/train_models.py` first.")
    else:
        left, right = st.columns([1, 1], gap="medium")
        with left, card("analyze_form"):
            ui.card_title("Email", "Subject and body are all the models need")
            default_subject, default_body, sample_key = "", "", "custom"
            if IS_DEMO:
                samples = get_demo_emails()
                options = ["✏️ Write my own email"] + [f"{e['sender_name']} — {e['subject']}" for e in samples]
                choice = st.selectbox("Start from a sample email (optional)", options)
                if choice != options[0]:
                    picked = samples[options.index(choice) - 1]
                    default_subject, default_body, sample_key = picked["subject"], picked["body"], picked["message_id"]

            with st.form("analyze_form", border=False):
                subject = st.text_input("Subject", value=default_subject, key=f"subject-{sample_key}",
                                        placeholder="e.g. Your flight departs tomorrow")
                body = st.text_area(
                    "Email body", value=default_body, height=220, key=f"body-{sample_key}",
                    placeholder="e.g. Dear customer, your flight to Mumbai departs tomorrow morning. Please check in online."
                )
                submitted = st.form_submit_button("Analyze email", type="primary", width="stretch")

        with right, card("analyze_result"):
            ui.card_title("Prediction", "Category, priority and a transparent 0–100 score")
            if not submitted:
                ui.empty_state("Results appear here after you click Analyze email.")
            elif not subject.strip() and not body.strip():
                st.warning("Please enter a subject or body to analyze.")
            else:
                try:
                    result = classify_email(subject, body)
                except ModelsNotTrainedError as exc:
                    st.error(str(exc))
                else:
                    conf = result["category_confidence"]
                    pconf = result["priority_confidence"]
                    ui.kpi_row([
                        ("Category", f"{ui.CATEGORY_ICON.get(result['category'], '')} {result['category']}",
                         f"{conf * 100:.0f}% confidence" if conf is not None else "confidence unavailable",
                         ui.ACCENT, "🏷️"),
                        ("Priority", ui.PRIORITY_LABEL[result["priority"]],
                         f"{pconf * 100:.0f}% confidence" if pconf is not None else "confidence unavailable",
                         ui.PRIORITY_COLOR[result["priority"]], "🎯"),
                    ])
                    breakdown = result["score_breakdown"]
                    st.markdown(f"**Priority score: {result['priority_score']} / 100**")
                    ui.score_gauge(result["priority_score"])
                    st.caption(
                        f"ML-derived base score {breakdown['ml_base_score']} "
                        f"+ rule-based adjustment {breakdown['rule_adjustment']:+d} = {breakdown['final_score']}"
                    )

                    st.markdown("**Why?**")
                    for reason in result["explanation"]:
                        st.markdown(f"- {reason}")

                    with st.expander("Engineered features used by the priority model"):
                        st.json(result["metadata_features"])


# -------------------------------------------------------------------------
# PAGE: ML Analytics
# -------------------------------------------------------------------------

elif page == "📊 ML Analytics":
    metrics = get_metrics() if models_available() else None
    ui.page_header(
        "ML Analytics",
        (f"Trained on {metrics['dataset_size']:,} synthetic emails · random seed {metrics['random_state']}"
         if metrics else ""),
        demo=IS_DEMO,
    )
    if not metrics:
        st.error("No metrics found. Please run `python src/train_models.py` first.")
    else:
        cat_best = metrics["category"]["results"][metrics["category"]["best_model"]]
        pri_best = metrics["priority"]["results"][metrics["priority"]["best_model"]]
        ui.kpi_row([
            ("Category model", metrics["category"]["best_model"], "selected by F1-score", ui.SERIES[0], "🏷️"),
            ("Category F1", f"{cat_best['f1_score']:.3f}", f"accuracy {cat_best['accuracy'] * 100:.1f}%",
             ui.SERIES[0], "📐"),
            ("Priority model", metrics["priority"]["best_model"], "selected by F1-score", ui.SERIES[1], "🎯"),
            ("Priority F1", f"{pri_best['f1_score']:.3f}", f"accuracy {pri_best['accuracy'] * 100:.1f}%",
             ui.SERIES[1], "📐"),
        ])

        tab_names = ["🏷️ Category model", "🎯 Priority model"] + (["🧪 Demo inbox evaluation"] if IS_DEMO else [])
        tabs = st.tabs(tab_names)

        for tab, (task_name, task_key) in zip(tabs, [("Category classification", "category"),
                                                     ("Priority classification", "priority")]):
            with tab:
                task = metrics[task_key]
                c1, c2 = st.columns([3, 2], gap="medium")
                with c1, card(f"cmp_{task_key}"):
                    ui.card_title(f"{task_name}: model comparison",
                                  f"Held-out test set · best model: {task['best_model']}")
                    ui.show_chart(chart_model_comparison(task), key=f"cmp_{task_key}")
                    rows = [{"Model": name, "Accuracy": res["accuracy"], "Precision": res["precision"],
                             "Recall": res["recall"], "F1-score": res["f1_score"]}
                            for name, res in task["results"].items()]
                    comparison_df = pd.DataFrame(rows).set_index("Model")
                    st.dataframe(comparison_df.style.format("{:.4f}").highlight_max(axis=0, color="#d6ecd6"),
                                 width="stretch")
                with c2, card(f"cm_{task_key}"):
                    best_res = task["results"][task["best_model"]]
                    ui.card_title("Confusion matrix", task["best_model"])
                    ui.show_chart(chart_confusion(best_res["confusion_matrix"], best_res["labels"]),
                                  key=f"cm_{task_key}")

        if IS_DEMO:
            with tabs[2]:
                st.caption(
                    "The sample inbox was never used for training. Each sample email has a human-assigned "
                    "reference label; here the live ML predictions are compared against those labels."
                )
                demo_df = prepare_dataframe(load_demo_emails(use_models=True))
                if (demo_df["prediction_source"] != "ML model").any():
                    st.warning("Could not run the ML models on the demo inbox.")
                else:
                    present_cats = [c for c in CATEGORIES
                                    if c in set(demo_df["expected_category"]) | set(demo_df["category"])]
                    cat_eval = evaluate_predictions(demo_df["expected_category"], demo_df["category"], present_cats)
                    pri_eval = evaluate_predictions(demo_df["expected_priority"], demo_df["priority"], PRIORITIES)

                    ui.kpi_row([
                        ("Category accuracy", f"{cat_eval['accuracy'] * 100:.1f}%", f"{len(demo_df)} sample emails",
                         ui.SERIES[0], "🏷️"),
                        ("Category F1 (macro)", f"{cat_eval['f1_score']:.3f}", "vs reference labels",
                         ui.SERIES[0], "📐"),
                        ("Priority accuracy", f"{pri_eval['accuracy'] * 100:.1f}%", f"{len(demo_df)} sample emails",
                         ui.SERIES[1], "🎯"),
                        ("Priority F1 (macro)", f"{pri_eval['f1_score']:.3f}", "vs reference labels",
                         ui.SERIES[1], "📐"),
                    ])
                    c1, c2 = st.columns(2, gap="medium")
                    with c1, card("demo_cm_cat"):
                        ui.card_title("Category", "Reference label vs ML prediction")
                        ui.show_chart(chart_confusion(cat_eval["confusion_matrix"], present_cats, "Reference"),
                                      key="demo_cm_cat")
                    with c2, card("demo_cm_pri"):
                        ui.card_title("Priority", "Reference label vs ML prediction")
                        ui.show_chart(chart_confusion(pri_eval["confusion_matrix"], PRIORITIES, "Reference"),
                                      key="demo_cm_pri")

                    mismatches = demo_df[(demo_df["category"] != demo_df["expected_category"])
                                         | (demo_df["priority"] != demo_df["expected_priority"])]
                    with card("mismatch"):
                        ui.card_title(f"Disagreements ({len(mismatches)})",
                                      "Emails where the model's prediction differs from the reference label")
                        st.dataframe(pd.DataFrame({
                            "Subject": mismatches["subject"].values,
                            "Reference category": mismatches["expected_category"].values,
                            "Predicted category": mismatches["category"].values,
                            "Reference priority": mismatches["expected_priority"].values,
                            "Predicted priority": mismatches["priority"].values,
                            "Score": mismatches["priority_score"].values,
                        }), hide_index=True, width="stretch")

        st.info(
            "**Model roles in this project:**\n\n"
            "- **Used for model comparison:** Naive Bayes, Logistic Regression, Decision Tree, Random Forest\n"
            "- **Final category model:** automatically selected above (highest F1-score)\n"
            "- **Final priority model:** automatically selected above (highest F1-score)\n"
            "- **Unsupervised clustering:** K-Means + PCA (see 'Email Clusters' page) - a separate, "
            "unsupervised component not used for the final category/priority predictions."
        )


# -------------------------------------------------------------------------
# PAGE: Email Clusters
# -------------------------------------------------------------------------

elif page == "🔬 Email Clusters":
    ui.page_header("Email Clusters", "Unsupervised grouping of your inbox by text similarity", demo=IS_DEMO)

    with st.expander("How is this different from classification?", expanded=False):
        st.markdown(
            "- **Classification** (category/priority models) = *supervised learning*: the model learns "
            "from emails that already have a known correct label.\n"
            "- **K-Means** = *unsupervised learning*: it groups emails purely by text similarity, with **no labels** involved.\n"
            "- **PCA** (Principal Component Analysis) = *dimensionality reduction*: it compresses hundreds of "
            "TF-IDF dimensions down to 2, purely so the clusters can be plotted on a 2D chart."
        )

    df = get_active_dataframe()
    if df.empty:
        no_emails_warning()
    else:
        n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=6)
        result = run_clustering(df["subject"].fillna("").tolist(), df["body"].fillna("").tolist(), n_clusters=n_clusters)

        if result is None:
            st.warning("Not enough emails to form meaningful clusters. Load more emails first.")
        else:
            plot_df = pd.DataFrame(result["pca_coords"], columns=["PC1", "PC2"])
            plot_df["ClusterId"] = result["cluster_labels"]
            plot_df["Cluster"] = [f"Cluster {c}" for c in result["cluster_labels"]]
            plot_df["Subject"] = df["subject"].fillna("").values
            plot_df["Category"] = df["category"].fillna("Unknown").values if "category" in df else "Unknown"

            c1, c2 = st.columns([3, 2], gap="medium")
            with c1, card("scatter"):
                ev = result["explained_variance"]
                ui.card_title("K-Means clusters (PCA 2D view)",
                              f"PCA keeps {sum(ev) * 100:.1f}% of variance "
                              f"({ev[0] * 100:.1f}% + {ev[1] * 100:.1f}%) · hover a point to see the email")
                fig = go.Figure()
                for cid in sorted(plot_df["ClusterId"].unique()):
                    part = plot_df[plot_df["ClusterId"] == cid]
                    fig.add_scatter(
                        x=part["PC1"], y=part["PC2"], mode="markers", name=f"Cluster {cid}",
                        marker=dict(size=11, color=ui.SERIES[cid % len(ui.SERIES)],
                                    symbol=ui.SYMBOLS[cid % len(ui.SYMBOLS)], line=dict(color="#ffffff", width=1.5)),
                        customdata=part[["Subject", "Category"]].values,
                        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · " + f"Cluster {cid}<extra></extra>",
                    )
                fig.update_xaxes(title="PC1", zeroline=False)
                fig.update_yaxes(title="PC2", zeroline=False)
                ui.show_chart(ui.style_fig(fig, height=430), key="scatter")
            with c2, card("cluster_cat"):
                ui.card_title("Clusters vs predicted categories",
                              "K-Means never saw these labels — overlap shows it found real structure")
                cluster_vs_cat = pd.crosstab(plot_df["Cluster"], plot_df["Category"])
                fig2 = go.Figure(go.Heatmap(
                    z=cluster_vs_cat.values, x=list(cluster_vs_cat.columns), y=list(cluster_vs_cat.index),
                    colorscale=BLUES, showscale=False, xgap=2, ygap=2, texttemplate="%{text}",
                    text=[[str(v) if v else "" for v in row] for row in cluster_vs_cat.values],
                    hovertemplate="%{y} · %{x}: %{z} emails<extra></extra>",
                ))
                fig2.update_yaxes(autorange="reversed", showgrid=False)
                fig2.update_xaxes(showgrid=False, tickangle=-40)
                ui.show_chart(ui.style_fig(fig2, height=430, legend=False), key="cluster_cat")

            with card("cluster_samples"):
                ui.card_title("Sample emails per cluster")
                cols = st.columns(min(3, len(set(result["cluster_labels"]))))
                for i, cluster_id in enumerate(sorted(set(result["cluster_labels"]))):
                    sample = plot_df[plot_df["ClusterId"] == cluster_id]
                    with cols[i % len(cols)]:
                        top_cat = sample["Category"].mode().iat[0] if not sample.empty else "—"
                        st.markdown(f"**Cluster {cluster_id}** · {len(sample)} emails · mostly *{top_cat}*")
                        for subj in sample["Subject"].head(5):
                            st.markdown(f"- {escape(subj)}")


# -------------------------------------------------------------------------
# PAGE: Gmail Settings
# -------------------------------------------------------------------------

elif page == "⚙️ Gmail Settings":
    ui.page_header("Gmail Settings", "Connect, sync and manage your local Gmail cache", demo=IS_DEMO)

    if IS_DEMO:
        st.info(
            "**Demo Mode is ON.** The dashboard is showing the built-in sample inbox, so no Gmail "
            "connection is needed. Turn **Demo Mode** OFF in the sidebar to view emails synced "
            "from your Gmail account with the controls below."
        )

    creds_exist = credentials_file_exists()
    authed = is_authenticated()
    stats = db.get_stats()

    ui.kpi_row([
        ("Connection", "Connected" if authed else "Not connected", "read-only Gmail scope",
         ui.PRIORITY_COLOR["LOW"] if authed else "#898781", "🔗"),
        ("Cached emails", stats["total_emails"], "stored locally in SQLite", ui.ACCENT, "💾"),
        ("Last sync", stats["last_synced"][:16].replace("T", " ") if stats["last_synced"] else "Never",
         "most recent Gmail fetch", "#52514e", "🕒"),
    ])

    if not creds_exist and IS_DEMO:
        st.caption("credentials.json has not been added yet - it is only needed when Demo Mode is OFF.")
    elif not creds_exist:
        st.error(
            "**credentials.json not found.** Gmail Mode will not work until you add it. "
            "See the README for step-by-step Google Cloud setup instructions, then place "
            "credentials.json in the InboxIQ project root."
        )

    c1, c2 = st.columns([2, 3], gap="medium")
    with c1, card("sync"):
        ui.card_title("Sync", "Fetch new emails and classify them")
        fetch_limit = st.selectbox("Number of emails to fetch", [25, 50, 100, 250, 500], index=1)

        if st.button("🔄 Sync Gmail", type="primary", width="stretch"):
            if not models_available():
                st.error("Please train the ML models first: `python src/train_models.py`")
            else:
                with st.spinner("Connecting to Gmail and fetching emails (read-only)..."):
                    try:
                        added, already_cached = sync_gmail(fetch_limit)
                        st.success(f"Synced successfully. {added} new email(s) fetched and classified. "
                                   f"{already_cached} email(s) were already cached and skipped.")
                        st.cache_data.clear()
                    except CredentialsNotFoundError as exc:
                        st.error(str(exc))
                    except GmailAuthError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"An unexpected error occurred while syncing Gmail: {exc}")

        if st.button("🗑 Clear Local Cache", width="stretch"):
            db.clear_all_data()
            st.cache_data.clear()
            st.success("Local InboxIQ cache cleared. Your Gmail account was not affected - "
                       "this only removed data from InboxIQ's local SQLite database.")

    with c2, card("cached_breakdown"):
        ui.card_title("Category breakdown (cached)", "Emails stored in the local Gmail cache")
        if stats["by_category"]:
            cached = pd.Series(stats["by_category"]).sort_values()
            fig = go.Figure(go.Bar(x=cached.values, y=list(cached.index), orientation="h", marker_color=ui.ACCENT,
                                   text=cached.values, textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}: %{x} emails<extra></extra>"))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(showgrid=False)
            ui.show_chart(ui.style_fig(fig, height=max(220, 28 * len(cached)), legend=False), key="cached")
        else:
            ui.empty_state("No cached emails yet.")


# -------------------------------------------------------------------------
# PAGE: About
# -------------------------------------------------------------------------

elif page == "ℹ️ About":
    ui.page_header("About InboxIQ", "Machine Learning Lab mini-project", demo=IS_DEMO)

    with card("about"):
        st.markdown("""
### Project Objective
InboxIQ is a Machine Learning Lab mini-project that connects to a real Gmail inbox
(read-only) and uses traditional, explainable ML algorithms to automatically
**classify emails into categories** and **predict their priority**, helping users
focus on what matters most in a busy inbox.

### Machine Learning Algorithms Used
- **Naive Bayes, Logistic Regression, Decision Tree, Random Forest** - compared
  for both category classification and priority classification; the best model
  (by F1-score) is automatically selected for each task.
- **TF-IDF (Term Frequency - Inverse Document Frequency)** - converts email text
  into numeric feature vectors.
- **K-Means + PCA** - an additional *unsupervised* component that groups emails
  by text similarity without using any labels, visualized in 2D.

### Demo Mode
With **Demo Mode** ON (sidebar toggle, on by default) InboxIQ shows a built-in
inbox of 45 fictional sample emails covering education, work, finance, shopping,
security, personal, promotions, social, travel, events and newsletters. Every
sample email is classified live by the trained ML models, completely offline -
no Gmail account, OAuth or internet connection is needed. All names, companies,
transactions and messages in the sample inbox are fictional.

### Gmail API Integration
InboxIQ uses the official Gmail API with OAuth 2.0 and the
`gmail.readonly` scope **only**. It can read email metadata and content, but
it can never send, delete, modify, archive, or label any email.

### Privacy
Your Gmail data is accessed only after Google OAuth authorization. InboxIQ
uses read-only Gmail access and does not send, delete, modify, archive, or
label emails. Email data is processed locally by this application and
cached locally in SQLite (`data/inboxiq.db`) on your own machine.

### Limitations
- The ML models are trained on a synthetic dataset, not on real personal emails,
  so real-world classification accuracy may vary from the training metrics.
- Only plain-text and HTML email bodies are analyzed; attachments are not processed.
- Priority scoring blends an ML-derived probability score with a small,
  clearly-labeled rule-based adjustment - it is not a perfect measure of
  true urgency.

### Future Enhancements
Gmail label integration, Outlook integration, spam detection, duplicate
detection, email summarization, personalized priority learning (from user
feedback), multilingual classification, attachment analysis, smart
notifications, model retraining pipelines, and cloud deployment.
        """)
