import numpy as np

from src.metrics import classification_metrics, per_attack_metrics


def test_confusion_counts_specificity_and_fpr():
    y = np.array([1, 1, 0, 0])
    yhat = np.array([1, 0, 1, 0])
    result = classification_metrics(y, yhat, np.array([0.9, 0.7, 0.8, 0.1]))
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)
    assert result["specificity"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert result["positive_prevalence"] == 0.5
    assert result["average_precision_auprc"] is not None


def test_per_attack_positive_recall_is_available_for_positive_only_family():
    y = np.array([1, 1, 0, 0])
    yhat = np.array([1, 0, 1, 0])
    families = np.array(["gear", "gear", "normal", "normal"])
    result = per_attack_metrics(y, yhat, np.array([0.9, 0.7, 0.8, 0.1]), families)
    assert result["gear"]["positive_window_recall"] == 0.5
    assert result["gear"]["precision"] is None
    assert result["normal"]["positive_window_recall"] is None
