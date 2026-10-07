"""
InboxIQ - ML Training Pipeline
=================================
Trains and evaluates the category classifier and the priority classifier
on the synthetic dataset (data/training_emails.csv), compares multiple
scikit-learn algorithms, automatically selects the best model for each
task, and saves everything needed for live prediction.

Run:
    python src/train_models.py

Outputs:
    models/category_model.pkl
    models/category_vectorizer.pkl
    models/priority_model.pkl
    models/priority_vectorizer.pkl   (dict: tfidf + scaler + category list)
    models/model_metrics.json        (real metrics for the ML Analytics page)
"""

import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_for_vectorizer, extract_metadata_features
from src.evaluation import evaluate_predictions, pick_best_model

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "training_emails.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CATEGORIES = [
    "Education", "Work", "Finance", "Shopping", "Travel", "Social",
    "Security", "Promotions", "Personal", "Events", "Newsletter", "Other",
]
PRIORITIES = ["LOW", "MEDIUM", "HIGH"]

RANDOM_STATE = 42


def ensure_training_data():
    """Generate the synthetic dataset automatically if it doesn't exist yet."""
    if os.path.exists(DATA_PATH):
        return
    print("Training data not found. Generating it now with data/generate_training_data.py ...")
    generator_path = os.path.join(PROJECT_ROOT, "data", "generate_training_data.py")
    if not os.path.exists(generator_path):
        raise FileNotFoundError(
            "data/generate_training_data.py is missing. Cannot create training data. "
            "Please restore this file and re-run training."
        )
    subprocess.run([sys.executable, generator_path], check=True, cwd=PROJECT_ROOT)
    if not os.path.exists(DATA_PATH):
        raise RuntimeError("Training data generation finished but training_emails.csv was still not found.")


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    required_cols = {"email_id", "subject", "body", "category", "priority"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")
    df = df.dropna(subset=["category", "priority"]).reset_index(drop=True)
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["clean_text"] = df.apply(lambda r: preprocess_for_vectorizer(r["subject"], r["body"]), axis=1)
    return df


# -----------------------------------------------------------------------
# CATEGORY MODEL TRAINING (supervised, text-only via TF-IDF)
# -----------------------------------------------------------------------

def train_category_models(df):
    print("\n=== Training Category Classification Models ===")

    X_text = df["clean_text"]
    y = df["category"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)  # fit ONLY on training data (no leakage)
    X_test = vectorizer.transform(X_test_text)

    candidate_models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=30),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    results = {}
    trained_models = {}
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, labels=CATEGORIES)
        results[name] = metrics
        trained_models[name] = model
        print(f"  {name:22s} | Accuracy={metrics['accuracy']:.4f}  F1={metrics['f1_score']:.4f}")

    best_name = pick_best_model(results)
    best_model = trained_models[best_name]
    print(f"Best category model: {best_name}")

    return {
        "vectorizer": vectorizer,
        "best_model_name": best_name,
        "best_model": best_model,
        "results": results,
    }


# -----------------------------------------------------------------------
# PRIORITY MODEL TRAINING (supervised, text + metadata + category features)
# -----------------------------------------------------------------------

def build_priority_features(df, tfidf_vectorizer=None, scaler=None, fit=False):
    """
    Build the combined feature matrix for priority prediction:
        [ TF-IDF text features | scaled metadata features | one-hot category ]
    If fit=True, fits a new tfidf_vectorizer/scaler; otherwise reuses the
    ones passed in (used for the held-out test split to avoid data leakage).
    """
    if fit:
        tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", min_df=2)
        text_matrix = tfidf_vectorizer.fit_transform(df["clean_text"])
    else:
        text_matrix = tfidf_vectorizer.transform(df["clean_text"])

    meta_rows = [extract_metadata_features(r["subject"], r["body"]) for _, r in df.iterrows()]
    meta_df = pd.DataFrame(meta_rows)
    meta_cols = list(meta_df.columns)

    if fit:
        scaler = MinMaxScaler()
        meta_scaled = scaler.fit_transform(meta_df.values)
    else:
        meta_scaled = scaler.transform(meta_df.values)

    category_onehot = pd.get_dummies(df["category"]).reindex(columns=CATEGORIES, fill_value=0).values

    combined = hstack([
        text_matrix,
        csr_matrix(meta_scaled),
        csr_matrix(category_onehot.astype(float)),
    ]).tocsr()

    return combined, tfidf_vectorizer, scaler, meta_cols


def train_priority_models(df):
    print("\n=== Training Priority Classification Models ===")

    y = df["priority"]
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_train, tfidf_vectorizer, scaler, meta_cols = build_priority_features(train_df, fit=True)
    X_test, _, _, _ = build_priority_features(test_df, tfidf_vectorizer=tfidf_vectorizer, scaler=scaler, fit=False)

    candidate_models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=30),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    results = {}
    trained_models = {}
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, labels=PRIORITIES)
        results[name] = metrics
        trained_models[name] = model
        print(f"  {name:22s} | Accuracy={metrics['accuracy']:.4f}  F1={metrics['f1_score']:.4f}")

    best_name = pick_best_model(results)
    best_model = trained_models[best_name]
    print(f"Best priority model: {best_name}")

    priority_vectorizer_bundle = {
        "tfidf": tfidf_vectorizer,
        "scaler": scaler,
        "categories": CATEGORIES,
        "meta_columns": meta_cols,
    }

    return {
        "vectorizer_bundle": priority_vectorizer_bundle,
        "best_model_name": best_name,
        "best_model": best_model,
        "results": results,
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    ensure_training_data()
    df = load_dataset()
    print(f"Loaded {len(df)} training emails from {DATA_PATH}")

    category_result = train_category_models(df)
    priority_result = train_priority_models(df)

    joblib.dump(category_result["best_model"], os.path.join(MODELS_DIR, "category_model.pkl"))
    joblib.dump(category_result["vectorizer"], os.path.join(MODELS_DIR, "category_vectorizer.pkl"))
    joblib.dump(priority_result["best_model"], os.path.join(MODELS_DIR, "priority_model.pkl"))
    joblib.dump(priority_result["vectorizer_bundle"], os.path.join(MODELS_DIR, "priority_vectorizer.pkl"))

    metrics_payload = {
        "category": {
            "best_model": category_result["best_model_name"],
            "results": category_result["results"],
            "labels": CATEGORIES,
        },
        "priority": {
            "best_model": priority_result["best_model_name"],
            "results": priority_result["results"],
            "labels": PRIORITIES,
        },
        "dataset_size": len(df),
        "random_state": RANDOM_STATE,
    }

    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n=== Training complete ===")
    print(f"Category model saved -> models/category_model.pkl ({category_result['best_model_name']})")
    print(f"Priority model saved -> models/priority_model.pkl ({priority_result['best_model_name']})")
    print("Metrics saved -> models/model_metrics.json")


if __name__ == "__main__":
    main()
