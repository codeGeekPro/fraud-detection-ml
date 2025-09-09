"""
Service de prédiction pour la détection de fraudes.
"""

from typing import Dict, Any, Tuple
import joblib
import logging
import yaml
import os

class PredictionService:
    """
    Service d'inférence pour la détection de fraudes.

    Args:
        config_path (str): Chemin vers le fichier de configuration YAML.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialise le service de prédiction en chargeant le modèle et les paramètres.

        Args:
            config_path (str): Chemin vers le fichier de configuration YAML.
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_info = {}
        self._load_config(config_path)
        self._load_model()

    def _load_config(self, config_path: str) -> None:
        """
        Charge la configuration depuis un fichier YAML.

        Args:
            config_path (str): Chemin du fichier YAML.
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self.model_path = config.get("model_path", "models/trained/model.joblib")
            self.model_info = config.get("model_info", {})
            self.logger.info(f"Configuration chargée depuis {config_path}")
        except Exception as e:
            self.logger.error(f"Erreur chargement config: {e}")
            raise

    def _load_model(self) -> None:
        """
        Charge le modèle ML depuis le chemin spécifié.
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Modèle chargé depuis {self.model_path}")
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle: {e}")
            raise

    def predict(self, features: Dict[str, Any]) -> Tuple[int, float]:
        """
        Prédit la classe et la probabilité pour une transaction.

        Args:
            features (Dict[str, Any]): Dictionnaire des features.

        Returns:
            Tuple[int, float]: (classe prédite, probabilité de fraude)
        """
        try:
            X = self._features_to_array(features)
            pred = int(self.model.predict(X)[0])
            proba = float(self.model.predict_proba(X)[0][1])
            self.logger.info(f"Prédiction: {features} => {pred}, {proba}")
            return pred, proba
        except Exception as e:
            self.logger.error(f"Erreur prédiction: {e}")
            raise

    def predict_proba(self, features: Dict[str, Any]) -> Tuple[int, float]:
        """
        Retourne la probabilité de fraude pour une transaction.

        Args:
            features (Dict[str, Any]): Dictionnaire des features.

        Returns:
            Tuple[int, float]: (classe prédite, probabilité de fraude)
        """
        return self.predict(features)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations du modèle.

        Returns:
            Dict[str, Any]: Infos du modèle (type, params, etc.)
        """
        return self.model_info

    def _features_to_array(self, features: Dict[str, Any]):
        """
        Transforme le dictionnaire de features en array pour le modèle.

        Args:
            features (Dict[str, Any]): Dictionnaire des features.

        Returns:
            array: Features sous forme de tableau 2D.
        """
        import numpy as np
        return np.array([list(features.values())])
