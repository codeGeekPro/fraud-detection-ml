"""XGBoost model implementation."""

from typing import Optional, Dict, Any
from .base_model import BaseModel
import xgboost as xgb


class XGBoostModel(BaseModel):
    """
    Implémentation du modèle XGBoost pour la détection de fraudes.

    Args:
        params (Optional[Dict[str, Any]]): Dictionnaire d'hyperparamètres
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise le modèle XGBoost avec les hyperparamètres fournis.

        Args:
            params (Optional[Dict[str, Any]]): Hyperparamètres
        """
        import logging

        self.logger = logging.getLogger(__name__)
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "scale_pos_weight": 1,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X, y) -> None:
        """
        Entraîne le modèle XGBoost.

        Args:
            X: Features d'entraînement
            y: Labels d'entraînement
        """
        self.logger.info("Entraînement du modèle XGBoost...")
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X) -> Any:
        """
        Prédit la classe des observations.

        Args:
            X: Features à prédire

        Returns:
            Any: Prédictions
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant la prédiction.")
        return self.model.predict(X)

    def predict_proba(self, X) -> Any:
        """
        Retourne les probabilités de prédiction.

        Args:
            X: Features à prédire

        Returns:
            Any: Probabilités
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant la prédiction.")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle entraîné.

        Args:
            path (str): Chemin du fichier
        """
        import joblib

        joblib.dump(self.model, path)
        self.logger.info(f"Modèle XGBoost sauvegardé dans {path}")

    def load(self, path: str) -> None:
        """
        Charge un modèle sauvegardé.

        Args:
            path (str): Chemin du fichier
        """
        import joblib

        self.model = joblib.load(path)
        self.is_fitted = True
        self.logger.info(f"Modèle XGBoost chargé depuis {path}")
