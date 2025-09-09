"""
Evaluator module for fraud detection.

Classe d'évaluation complète : toutes les métriques, visualisations, rapports automatiques, SHAP, comparaison de modèles.
Respecte PEP8, docstrings Google, type hints, logging, Black.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
import shap
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
import plotly.graph_objects as go
import plotly.express as px

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Evaluator:
    def compare_models(
        self, models: Dict[str, Any], X: pd.DataFrame, y: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare plusieurs modèles sur les métriques principales.

        Args:
            models (Dict[str, Any]): Dictionnaire {nom: modèle}
            X (pd.DataFrame): Features
            y (np.ndarray): Labels

        Returns:
            pd.DataFrame: Tableau comparatif des métriques
        """
        from src.utils.metrics import calculate_metrics

        results = []
        for name, model in models.items():
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            metrics = calculate_metrics(y, y_pred, y_proba)
            metrics["model"] = name
            results.append(metrics)
        df = pd.DataFrame(results).set_index("model")
        logger.info("Comparaison multi-modèles réalisée.")
        return df

    def shap_summary_plot(self, model, X: pd.DataFrame) -> None:
        """
        Affiche le summary plot SHAP pour l'interprétabilité globale.

        Args:
            model: Modèle entraîné
            X (pd.DataFrame): Features
        """
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show=True)

    """
    Classe d'évaluation pour la détection de fraudes.

    Args:
        y_true (np.ndarray): Labels réels
        y_pred (np.ndarray): Prédictions
        y_proba (np.ndarray): Probabilités
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def get_metrics(self) -> Dict[str, float]:
        """
        Calcule toutes les métriques prioritaires.
        Returns:
            Dict[str, float]: Dictionnaire des métriques
        """
        metrics = {
            "precision": precision_score(self.y_true, self.y_pred),
            "recall": recall_score(self.y_true, self.y_pred),
            "f1": f1_score(self.y_true, self.y_pred),
            "f0.5": f1_score(self.y_true, self.y_pred, beta=0.5),
            "f2": f1_score(self.y_true, self.y_pred, beta=2),
            "cohen_kappa": cohen_kappa_score(self.y_true, self.y_pred),
            "mcc": matthews_corrcoef(self.y_true, self.y_pred),
        }
        if self.y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(self.y_true, self.y_proba[:, 1])
            pr, rc, _ = precision_recall_curve(self.y_true, self.y_proba[:, 1])
            metrics["pr_auc"] = auc(rc, pr)
        return metrics

    def plot_confusion_matrix(self) -> go.Figure:
        """
        Affiche la matrice de confusion avec heatmap.
        Returns:
            go.Figure: Figure Plotly
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
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

    def plot_roc_curve(self) -> go.Figure:
        """
        Affiche la courbe ROC.
        Returns:
            go.Figure: Figure Plotly
        """
        if self.y_proba is None:
            raise ValueError("y_proba requis pour la courbe ROC")
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba[:, 1])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR")
        return fig

    def plot_pr_curve(self) -> go.Figure:
        """
        Affiche la courbe Precision-Recall.
        Returns:
            go.Figure: Figure Plotly
        """
        if self.y_proba is None:
            raise ValueError("y_proba requis pour la courbe PR")
        pr, rc, _ = precision_recall_curve(self.y_true, self.y_proba[:, 1])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name="PR"))
        fig.update_layout(
            title="Courbe Precision-Recall",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        return fig

    def plot_prediction_distribution(self) -> go.Figure:
        """
        Affiche la distribution des scores de prédiction.
        Returns:
            go.Figure: Figure Plotly
        """
        if self.y_proba is None:
            raise ValueError("y_proba requis pour la distribution")
        fig = px.histogram(
            self.y_proba[:, 1], nbins=50, title="Distribution des scores de prédiction"
        )
        return fig

    def plot_calibration(self) -> go.Figure:
        """
        Affiche le calibration plot.
        Returns:
            go.Figure: Figure Plotly
        """
        from sklearn.calibration import calibration_curve

        if self.y_proba is None:
            raise ValueError("y_proba requis pour la calibration")
        prob_true, prob_pred = calibration_curve(
            self.y_true, self.y_proba[:, 1], n_bins=10
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=prob_pred, y=prob_true, mode="lines+markers", name="Calibration"
            )
        )
        fig.update_layout(
            title="Calibration plot",
            xaxis_title="Score prédit",
            yaxis_title="Probabilité réelle",
        )
        return fig

    def plot_feature_importance(self, model, X: pd.DataFrame) -> go.Figure:
        """
        Affiche l'importance des features via SHAP.
        Args:
            model: Modèle entraîné
            X (pd.DataFrame): Features
        Returns:
            go.Figure: Figure Plotly
        """
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        fig = shap.plots.bar(shap_values, show=False)
        return fig

    def generate_report(self) -> Dict[str, Any]:
        """
        Génère un rapport complet avec toutes les métriques et visualisations.
        Returns:
            Dict[str, Any]: Rapport
        """
        report = {
            "metrics": self.get_metrics(),
            "confusion_matrix": self.plot_confusion_matrix(),
            "roc_curve": self.plot_roc_curve() if self.y_proba is not None else None,
            "pr_curve": self.plot_pr_curve() if self.y_proba is not None else None,
            "prediction_distribution": (
                self.plot_prediction_distribution()
                if self.y_proba is not None
                else None
            ),
            "calibration": (
                self.plot_calibration() if self.y_proba is not None else None
            ),
        }
        logger.info("Rapport d'évaluation généré.")
        return report
