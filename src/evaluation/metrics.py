"""
evaluation/metrics.py
---------------------
Compute and return all classification metrics as a dict.
Keeps evaluation logic fully separated from MLflow logging.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from src.utils.logger import logger


def compute_metrics(y_true, y_pred, class_names: list) -> dict:
    """
    Returns a flat dict of all metrics — ready to be passed to MLflow.

    Keys
    ----
    accuracy        : overall accuracy
    f1_weighted     : weighted F1 across all classes
    f1_<class>      : per-class F1 score (one key per class)
    """
    acc     = accuracy_score(y_true, y_pred)
    f1_w    = f1_score(y_true, y_pred, average="weighted")
    f1_each = f1_score(y_true, y_pred, average=None)
    report  = classification_report(y_true, y_pred, target_names=class_names)

    metrics = {"accuracy": acc, "f1_weighted": f1_w}
    for cls, score in zip(class_names, f1_each):
        metrics[f"f1_{cls}"] = score

    logger.info(f"Accuracy    : {acc:.4f}")
    logger.info(f"F1 weighted : {f1_w:.4f}")
    for cls, score in zip(class_names, f1_each):
        logger.info(f"F1 [{cls}]   : {score:.4f}")
    logger.info(f"\n{report}")

    return metrics
