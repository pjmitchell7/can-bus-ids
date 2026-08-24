"""Testable window-level metrics for the baseline evaluator."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _finite(value: float | int | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def classification_metrics(y: np.ndarray, yhat: np.ndarray, scores: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=np.int8)
    yhat = np.asarray(yhat, dtype=np.int8)
    if y.shape != yhat.shape:
        raise ValueError("y and yhat must have the same shape")
    tp = int(np.sum((y == 1) & (yhat == 1)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    positives = tp + fn
    negatives = tn + fp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    fpr = fp / negatives if negatives else None
    accuracy = (tp + tn) / len(y) if len(y) else None
    result = {
        "n_windows": int(len(y)),
        "positive_windows": int(positives),
        "negative_windows": int(negatives),
        "positive_prevalence": _finite(positives / len(y) if len(y) else None),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": _finite(accuracy),
        "specificity": _finite(specificity),
        "false_positive_rate": _finite(fpr),
        "precision": _finite(precision),
        "recall": _finite(recall),
        "f1": _finite(2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None),
        "flagged_window_rate": _finite(float(np.mean(yhat == 1)) if len(yhat) else None),
        "average_precision_auprc": None,
        "auroc": None,
    }
    if scores is not None and len(np.unique(y)) > 1:
        result["average_precision_auprc"] = _finite(average_precision_score(y, scores))
        result["auroc"] = _finite(roc_auc_score(y, scores))
    return result


def per_attack_metrics(
    y: np.ndarray,
    yhat: np.ndarray,
    scores: np.ndarray,
    attack_family: np.ndarray,
) -> dict[str, dict]:
    """Report exact family counts and positive recall without hiding one-class subsets."""
    result: dict[str, dict] = {}
    for family in sorted(set(np.asarray(attack_family).astype(str).tolist())):
        mask = np.asarray(attack_family).astype(str) == family
        family_scores = classification_metrics(y[mask], yhat[mask], scores[mask])
        positive_count = int(np.sum(y[mask] == 1))
        negative_count = int(np.sum(y[mask] == 0))
        if negative_count == 0:
            # A family subset with no standalone benign windows cannot define
            # precision or false-positive behavior within that subset.
            family_scores["precision"] = None
            family_scores["specificity"] = None
            family_scores["false_positive_rate"] = None
        positive_recall = (
            int(np.sum((y[mask] == 1) & (yhat[mask] == 1))) / positive_count
            if positive_count else None
        )
        family_scores["positive_window_recall"] = _finite(positive_recall)
        result[family] = family_scores
    return result
