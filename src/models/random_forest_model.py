"""Random Forest model implementation."""

from typing import Optional, Dict, Any
from .base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(BaseModel):
    """
    Implémentation du modèle Random Forest pour la détection de fraudes.

    Args:
        params (Optional[Dict[str, Any]]): Dictionnaire d'hyperparamètres
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise le modèle Random Forest avec les hyperparamètres fournis.

        Args:
            params (Optional[Dict[str, Any]]): Hyperparamètres
        """
        import logging

        self.logger = logging.getLogger(__name__)
        default_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": 42,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = RandomForestClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X, y) -> None:
        """
        Entraîne le modèle Random Forest.

        Args:
            X: Features d'entraînement
            y: Labels d'entraînement
        """
        self.logger.info("Entraînement du modèle Random Forest...")
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
        self.logger.info(f"Modèle Random Forest sauvegardé dans {path}")

    def load(self, path: str) -> None:
        """
        Charge un modèle sauvegardé.

        Args:
            path (str): Chemin du fichier
        """
        import joblib

        self.model = joblib.load(path)
        self.is_fitted = True
        self.logger.info(f"Modèle Random Forest chargé depuis {path}")
