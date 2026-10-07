# InboxIQ

**Intelligent Gmail Email Classification & Priority Management System**

A Machine Learning Lab mini-project that connects to a real Gmail inbox (read-only) and uses traditional, explainable scikit-learn algorithms to classify emails into categories and predict their priority, so you can focus on what matters most.

---

## Abstract

InboxIQ is a Streamlit web application that integrates with the Gmail API (read-only, OAuth 2.0) to fetch a user's real emails, then applies supervised machine learning (Naive Bayes, Logistic Regression, Decision Tree, Random Forest) to automatically classify each email into one of 12 categories and predict its priority (HIGH/MEDIUM/LOW) with a transparent 0–100 priority score. An unsupervised component (K-Means + PCA) additionally groups emails by text similarity for exploratory analysis. All predictions are cached locally in SQLite so the app never re-downloads or re-classifies the same email twice.

## Problem Statement

Modern inboxes receive dozens to hundreds of emails a day — security alerts, deadlines, promotions, newsletters — all mixed together with no automatic way to tell what actually needs attention right now. Gmail's own categorization (Primary/Social/Promotions) is coarse and does not rank emails by urgency.

## Motivation

As a student, important emails (exam schedules, interview calls, fee deadlines) often get buried under promotional and newsletter clutter. A lightweight, locally-run, privacy-respecting tool that automatically triages an inbox by category and urgency solves a real, everyday problem — and doubles as a complete, explainable ML Lab project covering text preprocessing, TF-IDF, multiple classification algorithms, model evaluation, and unsupervised clustering.

## Objectives

1. Connect to a real Gmail account using read-only OAuth 2.0 access.
2. Extract and preprocess email text (subject + body, including HTML emails).
3. Train and compare multiple ML classifiers for category and priority prediction.
4. Automatically select and persist the best-performing model for each task.
5. Generate a transparent, explainable 0–100 priority score.
6. Cache results locally in SQLite to avoid redundant Gmail calls.
7. Present everything in an interactive Streamlit dashboard with search, filters, and analytics.
8. Demonstrate unsupervised learning (K-Means + PCA) as a complementary technique.

## Features

- OAuth 2.0 Gmail sign-in with **read-only** access (`gmail.readonly` scope only)
- Fetches 25/50/100/250/500 emails, handles plain-text, multipart, and HTML bodies
- 12-category email classification: Education, Work, Finance, Shopping, Travel, Social, Security, Promotions, Personal, Events, Newsletter, Other
- 3-level priority prediction (HIGH/MEDIUM/LOW) with a 0–100 transparency-scored breakdown
- Model comparison: Naive Bayes vs Logistic Regression vs Decision Tree vs Random Forest, with real accuracy/precision/recall/F1 and confusion matrices
- K-Means + PCA unsupervised clustering with 2D visualization
- SQLite local cache — never re-fetches/re-classifies already-cached emails
- Full-text search plus category/priority filters and multiple sort orders
- **Demo Mode** — run and present the entire project without any Gmail connection at all
- Manual "Analyze Email" tool for testing arbitrary subject/body text
- Clear, honest error handling everywhere (missing credentials, expired tokens, missing models, etc.)

## System Architecture

```
                     ┌─────────────────────┐
                     │   Streamlit UI       │
                     │   (app.py)           │
                     └──────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐   ┌──────────▼──────────┐   ┌────────▼────────┐
│ gmail_service.py│   │    predict.py        │   │  clustering.py   │
│ OAuth + fetch   │   │ category+priority ML │   │  K-Means + PCA   │
│ (read-only)     │   │ pipeline             │   │  (unsupervised)  │
└───────┬─────────┘   └──────────┬───────────┘   └──────────────────┘
        │                        │
        │              ┌─────────▼──────────┐
        │              │  preprocessing.py   │
        │              │  text cleaning +    │
        │              │  TF-IDF + features  │
        │              └─────────┬──────────┘
        │                        │
┌───────▼────────────────────────▼───────┐
│           database.py (SQLite)          │
│           data/inboxiq.db               │
└──────────────────────────────────────────┘

Offline: data/generate_training_data.py → data/training_emails.csv
         → src/train_models.py → models/*.pkl + model_metrics.json
```

## Technologies

- **Python 3.10+**
- **Streamlit** — web UI
- **scikit-learn** — Naive Bayes, Logistic Regression, Decision Tree, Random Forest, TF-IDF, PCA, K-Means, metrics
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — charts
- **google-api-python-client / google-auth-oauthlib** — Gmail API + OAuth 2.0
- **BeautifulSoup4** — safe HTML email parsing
- **SQLite** (via Python's built-in `sqlite3`) — local cache
- **joblib** — model persistence

## Gmail API Integration

InboxIQ uses the official Gmail API with the `googleapiclient` Python client. It requests **only** the `https://www.googleapis.com/auth/gmail.readonly` scope. It never calls `messages.send`, `messages.trash`, `messages.modify`, or any label-mutating endpoint — see `src/gmail_service.py`.

## OAuth 2.0 Flow

1. On first Gmail Mode use, InboxIQ checks for `token.json`.
2. If absent (or invalid/expired without a refreshable token), it opens a local browser window via `InstalledAppFlow.run_local_server()` for you to sign in and grant **read-only** access.
3. The resulting credentials are saved to `token.json` for future runs.
4. If the token later expires, InboxIQ automatically attempts a silent refresh; if that fails, it asks you to re-authorize with a clear message instead of crashing.

## Dataset

Since real Gmail emails have no category/priority labels, InboxIQ trains on a **separate, synthetic, fictional dataset**: `data/generate_training_data.py` generates 1,800 reproducible fake emails (fixed random seed `42`), 150 per category across all 12 categories, using randomized templates (names, companies, amounts, dates). Priority labels are derived from a combination of each category's typical urgency plus the actual urgency/deadline/security keywords present in that specific email's generated text, with a little random noise — so the label is genuinely a function of content, not just a lookup table, which is what makes it learnable by a real ML model. **No real Gmail messages are ever used to build this dataset.**

## Data Preprocessing

`src/preprocessing.py` implements:
- HTML stripping (BeautifulSoup, no script execution)
- Lowercasing, URL removal, email-address removal, punctuation removal, whitespace normalization
- Subject + body combination (subject counted twice — it's usually more informative)
- Engineered numeric features: subject/body length, urgency/deadline/security keyword-hit counts, presence of `!`, presence of digits

## TF-IDF

`TfidfVectorizer` (scikit-learn) converts cleaned text into weighted term-frequency vectors. **It is fit only on the training split** and reused (via `.transform()`) on the test split and on all future predictions — this avoids data leakage.

## ML Algorithms

Compared for **both** category and priority classification:
- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

The model with the highest macro-F1 score (ties broken by accuracy) on a stratified 80/20 train/test split is automatically selected and saved for each task.

## Category Classification

Trained purely on TF-IDF text features (unigrams + bigrams, 5000 max features). Because the synthetic templates use highly category-distinctive vocabulary, all four models typically reach very high (often ~100%) test accuracy — this is expected and is explained in the ML Analytics page and Limitations section below.

## Priority Prediction

Trained on a **combined feature matrix**: TF-IDF text features + scaled engineered metadata features (lengths, urgency/deadline/security keyword hits) + one-hot encoded predicted category. This is a genuinely harder, more realistic 3-class problem (LOW/MEDIUM/HIGH) and typically reaches ~80% test accuracy — high enough to be useful, low enough to be honest.

The final **priority score (0–100)** shown in the UI is computed transparently in two clearly separated parts:
1. **ML-derived base score** — the model's class probabilities weighted against anchor points (LOW=15, MEDIUM=50, HIGH=85).
2. **Rule-based adjustment** — a small, capped (±10 max) nudge based on how many urgency/deadline/security signal words were detected.

Both numbers are shown separately in the "Prediction Details" panel — InboxIQ never claims a rule-based number is an ML prediction.

## K-Means

An **unsupervised** algorithm used only in the "Email Clusters" page. It groups emails purely by TF-IDF text similarity — it has no access to category or priority labels at all — to show a different, label-free way of discovering structure in an inbox.

## PCA

Principal Component Analysis reduces the (hundreds-of-dimensions) TF-IDF vectors down to 2 dimensions purely so the K-Means clusters can be plotted on a 2D scatter chart.

## SQLite Cache

`src/database.py` manages `data/inboxiq.db`, a single `emails` table keyed by `message_id`. Syncing Gmail only downloads and classifies messages whose id isn't already cached. The "🗑 Clear Local Cache" button deletes only this local table — it never touches Gmail.

## Streamlit UI

Seven pages: 🏠 Dashboard, 📥 Smart Inbox, 🤖 Analyze Email, 📊 ML Analytics, 🔬 Email Clusters, ⚙️ Gmail Settings, ℹ️ About — plus a sidebar **Demo Mode** ON/OFF toggle. The interface is styled like a modern mail client (avatars, unread markers, category and priority chips, priority-score bars, expandable email view) with interactive Plotly charts; all styling lives in `src/ui.py`.

---

## Installation

### Step 1 — Install Python

Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/) (check "Add Python to PATH" during setup on Windows).

### Step 2 — Get the project files

Copy the entire `InboxIQ/` folder onto your machine (e.g. `C:\Users\<you>\InboxIQ`).

### Step 3 — Create a virtual environment and install dependencies

Open PowerShell or Command Prompt inside the `InboxIQ` folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Google Cloud Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (e.g. "InboxIQ").
3. In the left menu, go to **APIs & Services → Library**, search for **Gmail API**, and click **Enable**.
4. Go to **APIs & Services → OAuth consent screen**. Choose **External** (unless you have a Workspace), fill in the required app name/support email, and add your own Gmail address as a **test user**.
5. Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**.
6. Select **Application type: Desktop app**, give it a name, and click **Create**.
7. Click **Download JSON** on the credential you just created.
8. Rename the downloaded file to exactly `credentials.json`.
9. Place `credentials.json` in the **root** of the `InboxIQ` project folder (next to `app.py`).

Official reference: https://developers.google.com/workspace/gmail/api/quickstart/python

> `credentials.json` and `token.json` are already listed in `.gitignore` — never commit them.

---

## Running the Project

### Step 4 — Generate the synthetic training dataset

```bash
python data\generate_training_data.py
```

### Step 5 — Train the ML models

```bash
python src\train_models.py
```

This prints accuracy/F1 for every model compared, tells you which model was selected for category vs priority, and saves everything into `models/`.

### Step 6 — Launch the app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Demo Mode

**Demo Mode is ON by default** — the sidebar toggle reads *Demo Mode: ON* and every page shows a small "Demo Mode — Showing Sample Emails" banner. The app works immediately, completely offline, with **no Gmail account, OAuth or credentials.json needed**.

- The demo inbox (`src/demo_data.py`) contains **45 fictional emails** — assignment deadlines, exam timetables, an internship interview and offer letter, bank debit alerts, order confirmations, a scholarship notice, password/login security alerts, project meeting reminders, promotions, social notifications, personal messages, travel and event emails. Each has a sender name and address, date/time, read/unread status and important/starred status. All people, companies, transactions and messages are invented.
- Every demo email is classified **live by the trained ML models** (category, priority, 0–100 priority score). Each email also carries a human-assigned reference label, used only on **📊 ML Analytics → Demo Inbox Evaluation** to measure how well the models agree with a human (and as a fallback if the models are not trained yet).
- The Dashboard shows total / unread / high-priority / important counts, category and priority distributions, priority-score statistics and histogram, category-vs-priority, read-vs-unread, emails over time, a "Needs Your Attention" list, recent emails and important emails.
- Demo emails are kept in memory only and are never written to the Gmail SQLite cache.

To start the app with Demo Mode OFF by default, set `INBOXIQ_DEMO_MODE=0` before launching (`set INBOXIQ_DEMO_MODE=0` in Command Prompt).

### Quick start on Windows

Double-click `setup.bat` once (creates `venv`, installs requirements, trains the models), then double-click `run.bat`.

## Gmail Mode

Switch **Demo Mode** OFF in the sidebar, go to **⚙️ Gmail Settings**, choose how many emails to fetch, and click **🔄 Sync Gmail**. The first time, a browser window will open asking you to sign in to Google and grant **read-only** access. After authorizing, your real inbox emails are fetched, classified, and cached locally, then shown across the app.

---

## Model Evaluation

Run `python src\train_models.py` to see live, real (not hard-coded) accuracy, precision, recall, and F1-score for all four algorithms on both tasks, plus which model was automatically selected. The same numbers are shown in the app's **📊 ML Analytics** page, along with confusion matrices.

## Results

In our test run on the bundled 1,800-email synthetic dataset (seed 42):
- **Category classification**: all four models reached very high accuracy (the synthetic category templates use highly distinctive vocabulary per category, so this is expected, not a bug — see Limitations).
- **Priority classification**: ~80–82% accuracy for the best model (Logistic Regression / Naive Bayes were closely competitive), a much harder and more realistic 3-class problem.

Exact numbers will vary slightly depending on scikit-learn version but the pipeline, evaluation, and model-selection logic are fully deterministic given the fixed random seed.

## Limitations

- Trained on synthetic, templated data — real-world Gmail text is messier and less lexically distinctive per category, so live accuracy on your real inbox will likely be lower than the training metrics.
- Only plain-text/HTML bodies are analyzed — attachments are not processed or downloaded.
- Priority scoring blends a genuine ML prediction with a small, clearly-labeled rule-based nudge — it is a helpful approximation of urgency, not a certified measure of it.
- No personalization: the model does not currently learn from a specific user's feedback/corrections.
- English-language emails only.

## Future Enhancements

Gmail label integration, Outlook/Office365 integration, spam detection, duplicate detection, automatic email summarization, personalized priority learning from user feedback, multilingual classification, attachment analysis, smart push notifications, in-app feedback loop for model retraining, and cloud deployment.

## Privacy

> Your Gmail data is accessed only after Google OAuth authorization. InboxIQ uses read-only Gmail access and does not send, delete, modify, archive, or label emails. Email data is processed locally by this application and cached locally in SQLite.

This is not an absolute security guarantee beyond what is described above — treat `token.json` and `credentials.json` like any other secret on your machine, and never commit them to version control.

## Conclusion

InboxIQ demonstrates a complete, honest, end-to-end machine learning pipeline — from raw text preprocessing through TF-IDF feature extraction, multi-algorithm supervised classification with real evaluation metrics, transparent score computation, and unsupervised clustering — wrapped in a genuinely useful, privacy-respecting, read-only Gmail productivity tool.

---

## Project Structure

```
InboxIQ/
├── app.py
├── requirements.txt
├── README.md
├── VIVA.md
├── PROJECT_REPORT.md
├── .gitignore
├── setup.bat                 # one-time Windows setup (venv + install + train)
├── run.bat                   # start the app on Windows
├── .streamlit/config.toml    # app theme
├── credentials.json          # YOU ADD THIS LOCALLY (see Google Cloud Setup)
├── data/
│   ├── generate_training_data.py
│   ├── training_emails.csv
│   └── inboxiq.db
├── models/
│   ├── category_model.pkl
│   ├── category_vectorizer.pkl
│   ├── priority_model.pkl
│   ├── priority_vectorizer.pkl
│   └── model_metrics.json
├── src/
│   ├── __init__.py
│   ├── gmail_service.py
│   ├── database.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── predict.py
│   ├── evaluation.py
│   ├── clustering.py
│   ├── demo_data.py          # built-in fictional demo inbox (Demo Mode)
│   └── ui.py                 # front-end components: styling, email list, KPI tiles, chart theme
└── tests/
    ├── test_preprocessing.py
    ├── test_database.py
    ├── test_prediction.py
    └── test_demo_data.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ML models are not trained yet` | Run `python src\train_models.py` |
| `credentials.json not found` | Follow the Google Cloud Setup section above |
| Gmail login window doesn't open | Check firewall isn't blocking `localhost`; re-run `streamlit run app.py` |
| `token.json` seems stuck / login loop | Delete `token.json` and try **🔄 Sync Gmail** again |
| Empty Smart Inbox in Gmail Mode | Go to **⚙️ Gmail Settings** and click **🔄 Sync Gmail** first |
