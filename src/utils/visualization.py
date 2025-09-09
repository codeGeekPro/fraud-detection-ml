"""Visualization utilities for fraud detection."""

from typing import Optional, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Crée une matrice de confusion interactive avec Plotly.

    Args:
        y_true (np.ndarray): Labels réels
        y_pred (np.ndarray): Prédictions

    Returns:
        go.Figure: Figure Plotly
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Réel", color="Nb"),
        x=["Non-Fraude", "Fraude"],
        y=["Non-Fraude", "Fraude"],
    )
    fig.update_layout(title="Matrice de confusion")
    return fig
