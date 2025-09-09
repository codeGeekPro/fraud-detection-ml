"""Custom metrics for fraud detection."""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calcule les métriques d'évaluation pour la détection de fraudes.

    Args:
        y_true (np.ndarray): Labels réels
        y_pred (np.ndarray): Prédictions
        y_proba (np.ndarray): Probabilités

    Returns:
        Dict[str, float]: Dictionnaire des métriques
    """
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        fbeta_score,
        cohen_kappa_score,
        matthews_corrcoef,
        roc_auc_score,
        precision_recall_curve,
        auc,
    )

    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f0.5": fbeta_score(y_true, y_pred, beta=0.5),
        "f2": fbeta_score(y_true, y_pred, beta=2),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba[:, 1]),
    }
    pr, rc, _ = precision_recall_curve(y_true, y_proba[:, 1])
    metrics["pr_auc"] = auc(rc, pr)
    return metrics
