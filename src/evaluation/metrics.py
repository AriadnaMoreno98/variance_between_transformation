import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score,
)


def compute_classification_metrics(y_true, y_pred):
    has_two_classes = len(np.unique(np.asarray(y_true))) >= 2

    if has_two_classes:
        roc_auc = roc_auc_score(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)
    else:
        roc_auc = None
        pr_auc = None

    recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    g_mean = np.sqrt(recall_0 * recall_1)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc,
        "g_mean": g_mean,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "pr_auc": pr_auc,
    }
