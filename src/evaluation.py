"""
InboxIQ - Model Evaluation Utilities
=======================================
Shared functions for computing real classification metrics (accuracy,
precision, recall, F1, confusion matrix) from actual model predictions.
No metric in this project is hard-coded - everything here is calculated
directly from y_true / y_pred arrays produced during training.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_predictions(y_true, y_pred, labels):
    """
    Compute accuracy/precision/recall/F1 (macro-averaged, suitable for
    multi-class problems with balanced-ish classes) and a confusion matrix.
    Returns a plain dict that is JSON-serializable (lists, not numpy arrays).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
        "classification_report": report,
    }


def pick_best_model(results_by_model, metric="f1_score"):
    """
    Given {model_name: metrics_dict}, return the name of the model with the
    highest value of `metric`. Ties broken by accuracy as a tiebreaker.
    """
    return max(
        results_by_model.items(),
        key=lambda item: (item[1][metric], item[1]["accuracy"]),
    )[0]
