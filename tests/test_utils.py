"""Tests pour les utilitaires d'évaluation et de visualisation."""

import numpy as np
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_confusion_matrix


def test_calculate_metrics():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_proba = np.array(
        [[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]
    )
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    assert "precision" in metrics
    assert "pr_auc" in metrics
    assert metrics["precision"] >= 0 and metrics["precision"] <= 1


def test_plot_confusion_matrix():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    fig = plot_confusion_matrix(y_true, y_pred)
    assert fig is not None
    assert hasattr(fig, "to_plotly_json")
