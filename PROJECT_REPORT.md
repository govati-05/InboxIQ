# PROJECT REPORT

## 1. Title

**InboxIQ – Intelligent Gmail Email Classification & Priority Management System**

## 2. Abstract

InboxIQ is a machine learning-based web application that connects to a real Gmail inbox using read-only OAuth 2.0 access, and automatically organizes incoming emails by classifying them into 12 meaningful categories and predicting their priority level (HIGH/MEDIUM/LOW) with a transparent 0–100 priority score. The system compares four supervised classification algorithms — Multinomial Naive Bayes, Logistic Regression, Decision Tree, and Random Forest — using real, measured evaluation metrics (accuracy, precision, recall, F1-score) and automatically selects the best-performing model for each task. An additional unsupervised component (K-Means clustering with PCA visualization) demonstrates label-free grouping of emails by text similarity. Results are cached locally using SQLite to avoid redundant Gmail API calls, and the entire system is presented through an interactive Streamlit dashboard.

## 3. Introduction

Email remains the primary communication channel for academic, professional, and personal matters, but the sheer volume of incoming messages — security alerts, deadlines mixed with promotions and newsletters — makes it hard to quickly identify what needs immediate attention. InboxIQ addresses this by applying supervised machine learning to automatically triage an inbox, while strictly respecting user privacy through read-only access and local-only data processing.

## 4. Problem Statement

Given a real Gmail inbox, automatically (a) categorize each email into one of several meaningful, human-understandable categories, and (b) predict how urgently the user should act on it, without requiring any manual labeling by the user, and without ever modifying the user's actual mailbox.

## 5. Existing System

Gmail's built-in "Categories" tab (Primary/Social/Promotions/Updates/Forums) provides broad topic bucketing but no urgency ranking, and its categories are not tailored to a student's specific needs (e.g. it does not distinguish "Education" deadlines from generic "Updates"). Generic priority inbox features (e.g. Gmail's "Important" marker) use opaque, non-explainable heuristics and cannot be inspected, compared, or retrained by the user.

## 6. Proposed System

InboxIQ proposes an open, explainable, locally-run alternative: a Streamlit application that (1) fetches real emails via the official Gmail API with strictly read-only access, (2) applies transparent, comparably-evaluated ML models trained on a reproducible synthetic dataset to assign both a category and a priority, (3) shows exactly how each prediction was derived (confidence scores, keyword signals, ML vs rule-based score breakdown), and (4) caches everything locally so the same email is never needlessly re-processed.

## 7. Objectives

- Integrate with the Gmail API using OAuth 2.0 and read-only access only.
- Build a reproducible synthetic training dataset covering 12 categories and 3 priority levels.
- Preprocess raw email text (including HTML) into clean, model-ready features.
- Train and fairly compare four supervised ML algorithms per task.
- Automatically select and persist the best model based on real test-set metrics.
- Produce a transparent, explainable 0–100 priority score.
- Cache all fetched/classified data locally using SQLite.
- Provide an unsupervised (K-Means + PCA) view of the inbox as a complementary analysis.
- Present all of the above through a usable, filterable, searchable Streamlit dashboard.
- Support a fully offline "Demo Mode" for presentation without Gmail access.

## 8. Requirements

### Functional Requirements
- OAuth 2.0 Gmail sign-in and token persistence/refresh
- Fetch configurable numbers of emails (25/50/100/250/500)
- Parse plain-text, multipart, and HTML email bodies safely
- Classify category and priority for each email
- Store/retrieve/search/filter emails in SQLite
- Manual single-email analysis tool
- Model comparison and metrics visualization
- Unsupervised clustering visualization

### Non-Functional Requirements
- Read-only Gmail access only (no send/delete/modify/label operations)
- No hard-coded or fabricated metrics — all evaluation numbers computed from real model predictions
- Graceful, user-readable error handling for all failure modes (missing credentials, expired tokens, missing models, network errors, empty inbox, etc.)
- Cross-platform (developed/tested on Windows 11), no Docker/WSL/Node.js required

### Software Requirements
Python 3.10+, Streamlit, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, google-api-python-client, google-auth-oauthlib, beautifulsoup4 (see `requirements.txt`).

## 9. System Architecture

```
Streamlit UI (app.py)
   │
   ├── gmail_service.py  → OAuth 2.0 + Gmail API (read-only fetch)
   ├── predict.py         → loads trained models, runs classify_email()
   ├── clustering.py      → K-Means + PCA (unsupervised)
   ├── preprocessing.py   → text cleaning, TF-IDF prep, feature extraction
   └── database.py        → SQLite local cache (data/inboxiq.db)

Offline training pipeline:
   data/generate_training_data.py → data/training_emails.csv
        → src/train_models.py → models/*.pkl, model_metrics.json
```

## 10. Dataset

`data/generate_training_data.py` produces 1,800 synthetic, fictional emails (fixed random seed 42), 150 per category across the 12 defined categories, using randomized subject/body templates (varying names, companies, cities, amounts, dates). Priority labels (HIGH/MEDIUM/LOW) are derived from a formula combining each category's typical baseline urgency with the actual urgency/deadline/security keywords detected in that specific generated email's text, plus small random noise — making the label a genuine function of content rather than a fixed per-category lookup, so it is learnable by a text-based classifier. No real Gmail messages are used anywhere in dataset construction.

## 11. Data Preprocessing

Implemented in `src/preprocessing.py`:
- HTML → plain text conversion via BeautifulSoup (scripts/styles stripped, no execution)
- Lowercasing, URL removal, email-address removal, non-alphanumeric character removal, whitespace normalization
- Subject and body combined into one text blob (subject duplicated for extra weight)
- Engineered numeric features: subject length, body length, urgency-keyword hit count, deadline-keyword hit count, security-keyword hit count, presence of `!`, presence of digits

## 12. Feature Extraction

- **Category model**: TF-IDF (unigrams + bigrams, up to 5,000 features) on the cleaned combined text — fit only on the training split.
- **Priority model**: TF-IDF (up to 3,000 features) + Min-Max-scaled engineered numeric features + one-hot encoded predicted category, concatenated into a single sparse feature matrix.

## 13. ML Algorithms

Four supervised algorithms are trained and compared for each of the two tasks:
1. **Multinomial Naive Bayes** — probabilistic, assumes conditional feature independence, fast baseline for text.
2. **Logistic Regression** — linear model estimating class probabilities via a softmax/sigmoid function.
3. **Decision Tree** — recursively splits on feature thresholds to reach a class decision.
4. **Random Forest** — an ensemble/bagging of many decision trees for improved generalization.

Unsupervised: **K-Means** clustering (on TF-IDF vectors) with **PCA** used purely for 2D visualization.

## 14. Training

`src/train_models.py`: loads/generates the dataset → cleans text → performs an 80/20 stratified train/test split → fits TF-IDF (and the metadata scaler, for priority) **only** on the training split → trains all four candidate models per task → evaluates each on the untouched test split → selects the best model per task by macro-F1 (tie-broken by accuracy) → persists the winning model, its vectorizer(s), and a full metrics JSON.

## 15. Evaluation

`src/evaluation.py` computes accuracy, macro-averaged precision/recall/F1, and a full confusion matrix directly from `y_test` vs `y_pred` using scikit-learn's metrics functions — no metric is hard-coded. In our reference run: category classification reached very high accuracy across all four models (explained by the lexically distinctive per-category synthetic templates), while priority classification — a harder, more nuanced 3-class problem — reached approximately 80–82% test accuracy for the best model.

## 16. Gmail Integration

`src/gmail_service.py` uses `google-auth-oauthlib`'s `InstalledAppFlow` for the OAuth 2.0 consent flow, requesting only the `gmail.readonly` scope. Once authorized, `googleapiclient.discovery.build("gmail", "v1", ...)` is used to list and fetch INBOX messages, with pagination support and per-message error tolerance (a single unreadable message is skipped rather than aborting the whole sync). Message payloads are recursively parsed to extract plain-text bodies, falling back to HTML-to-text conversion for HTML-only messages.

## 17. Database

`src/database.py` manages a single SQLite table (`emails`) in `data/inboxiq.db`, keyed by Gmail's `message_id` (primary key, enforcing no duplicates via `INSERT ... ON CONFLICT DO UPDATE`). It supports filtered/sorted/searched retrieval, aggregate statistics for the dashboard, and a full local-only clear operation that never touches the user's actual Gmail account.

## 18. UI

Built with Streamlit across seven pages (Dashboard, Smart Inbox, Analyze Email, ML Analytics, Email Clusters, Gmail Settings, About), plus a sidebar Demo Mode/Gmail Mode switch, search box, category/priority filters, and multiple sort orders.

## 19. Results

- Category classification: near-perfect test accuracy across all four compared algorithms on the synthetic dataset.
- Priority classification: ~80–82% test accuracy for the best-selected model (typically Logistic Regression or Naive Bayes), a realistic result for a genuinely content-dependent 3-class problem.
- The system successfully fetches, classifies, caches, and displays real Gmail emails end-to-end when authorized, and works fully offline in Demo Mode using the synthetic dataset.

## 20. Advantages

- Fully explainable: every prediction shows its confidence and the specific signals behind it.
- Privacy-respecting by design: read-only scope, local-only processing and storage.
- No vendor/API lock-in to paid AI services — runs entirely on open-source scikit-learn models.
- Reproducible: fixed random seeds throughout for consistent, gradeable results.
- Works with zero Gmail setup via Demo Mode, ideal for presentations.

## 21. Limitations

- Trained on synthetic, templated data, so real-world accuracy may differ from reported training metrics.
- Does not process email attachments.
- Priority score combines a genuine ML prediction with a small, clearly-labeled rule-based adjustment.
- No per-user personalization/feedback loop yet.
- English-language emails only.

## 22. Future Scope

Gmail label integration for two-way organization, Outlook/Office365 support, spam and duplicate detection, automatic email summarization, personalized priority learning from explicit user feedback, multilingual classification, attachment content analysis, smart notifications, a model-retraining feedback loop, and cloud deployment for multi-device access.

## 23. Conclusion

InboxIQ demonstrates that a small, well-structured, entirely open-source machine learning pipeline — careful text preprocessing, TF-IDF feature extraction, multi-algorithm supervised comparison with honest evaluation metrics, a transparent scoring formula, and a complementary unsupervised clustering view — can be assembled into a genuinely useful, privacy-conscious productivity tool, while remaining simple enough for a third-year CSE student to fully understand, explain, and defend in a viva.

## 24. References

1. Google Developers, "Gmail API Python Quickstart" — https://developers.google.com/workspace/gmail/api/quickstart/python
2. scikit-learn documentation — https://scikit-learn.org/stable/documentation.html
3. Streamlit documentation — https://docs.streamlit.io/
4. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR 12, 2011.
5. Python `sqlite3` documentation — https://docs.python.org/3/library/sqlite3.html
6. BeautifulSoup documentation — https://www.crummy.com/software/BeautifulSoup/bs4/doc/
