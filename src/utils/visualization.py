"""Visualization utilities for fraud detection."""
from typing import Optional, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Crée une matrice de confusion interactive."""
    pass  # À implémenter
