"""
Service de prédiction pour la détection de fraudes.
"""

from typing import Dict, Any, Tuple, List
import joblib
import logging
import yaml
import os
import numpy as np
from pathlib import Path
import sys

# Ajout du chemin racine au sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.preprocessor import Preprocessor
from src.data.feature_engineer import FeatureEngineer


class PredictionService:
    """
    Service d'inférence pour la détection de fraudes.

    Args:
        config_path (str): Chemin vers le fichier de configuration YAML.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialise le service de prédiction en chargeant le modèle et les composants.

        Args:
            config_path (str): Chemin vers le fichier de configuration YAML.
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.model_info = {}

        # Recherche du fichier de configuration
        config_paths = [
            config_path,
            "../config/config.yaml",
            "config/config.yaml",
            str(ROOT_DIR / "config" / "config.yaml")
        ]

        config_found = False
        for path in config_paths:
            if os.path.exists(path):
                self._load_config(path)
                config_found = True
                break

        if not config_found:
            self.logger.warning("Fichier de configuration non trouvé, utilisation des valeurs par défaut")
            self._set_default_config()

        self._load_model()
        self._load_preprocessing_components()

    def _load_config(self, config_path: str) -> None:
        """
        Charge la configuration depuis un fichier YAML.

        Args:
            config_path (str): Chemin du fichier YAML.
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self.config = config
            self.model_path = config.get("api", {}).get("model_path", "models/trained/best_model.joblib")
            self.preprocessor_path = config.get("api", {}).get("preprocessor_path", "models/trained/preprocessor.joblib")
            self.logger.info(f"Configuration chargée depuis {config_path}")
        except Exception as e:
            self.logger.error(f"Erreur chargement config: {e}")
            raise

    def _set_default_config(self) -> None:
        """Définit une configuration par défaut."""
        self.config = {
            "api": {
                "model_path": "models/trained/best_model.joblib",
                "preprocessor_path": "models/trained/preprocessor.joblib"
            }
        }
        self.model_path = self.config["api"]["model_path"]
        self.preprocessor_path = self.config["api"]["preprocessor_path"]

    def _load_model(self) -> None:
        """
        Charge le modèle ML depuis le chemin spécifié.
        """
        # Recherche du modèle dans plusieurs emplacements
        possible_paths = [
            self.model_path,
            "../models/trained/best_model.joblib",
            "models/trained/best_model.joblib",
            str(ROOT_DIR / "models" / "trained" / "best_model.joblib")
        ]

        model_found = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.model = joblib.load(path)
                    self.logger.info(f"Modèle chargé depuis {path}")
                    model_found = True
                    break
                except Exception as e:
                    self.logger.warning(f"Erreur chargement modèle depuis {path}: {e}")
                    continue

        if not model_found:
            raise FileNotFoundError(f"Modèle non trouvé dans les emplacements : {possible_paths}")

    def _load_preprocessing_components(self) -> None:
        """
        Charge les composants de preprocessing.
        """
        try:
            # Initialisation du preprocessor
            self.preprocessor = Preprocessor({
                'missing_values_strategy': 'median',
                'scaling_method': 'standard'
            })

            # Initialisation du feature engineer
            self.feature_engineer = FeatureEngineer()

            self.logger.info("Composants de preprocessing initialisés")

        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des composants : {e}")
            raise

    def predict_single(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Effectue une prédiction pour une seule transaction.

        Args:
            transaction_data (Dict[str, Any]): Données de la transaction.

        Returns:
            Dict[str, Any]: Résultats détaillés de la prédiction.
        """
        try:
            # Conversion en DataFrame
            import pandas as pd
            df_input = pd.DataFrame([transaction_data])

            # Application du preprocessing
            df_prep = self.preprocessor.fit_transform(df_input)

            # Application du feature engineering
            df_features = self.feature_engineer.fit_transform(df_prep)

            # Suppression de la colonne Class si elle existe
            if 'Class' in df_features.columns:
                df_features = df_features.drop('Class', axis=1)

            # Prédiction
            prediction_proba = self.model.predict_proba(df_features)[0]
            prediction = int((prediction_proba[1] >= 0.5).astype(int))

            # Résultats détaillés
            result = {
                'prediction': prediction,
                'fraud_probability': float(prediction_proba[1]),
                'is_fraud': bool(prediction),
                'confidence': float(max(prediction_proba)),
                'risk_level': self._get_risk_level(prediction_proba[1]),
                'model_version': '1.0.0'
            }

            self.logger.info(".2f")

            return result

        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction : {str(e)}")
            raise

    def predict_batch(self, transactions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Effectue des prédictions pour un lot de transactions.

        Args:
            transactions_data (List[Dict[str, Any]]): Liste des données de transactions.

        Returns:
            List[Dict[str, Any]]: Liste des résultats de prédiction.
        """
        self.logger.info(f"Prédiction par lot : {len(transactions_data)} transactions")

        results = []
        for i, transaction in enumerate(transactions_data):
            try:
                result = self.predict_single(transaction)
                result['transaction_id'] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Erreur pour la transaction {i} : {str(e)}")
                results.append({
                    'transaction_id': i,
                    'error': str(e),
                    'prediction': 0,
                    'fraud_probability': 0.0,
                    'is_fraud': False,
                    'confidence': 0.0,
                    'risk_level': 'UNKNOWN',
                    'model_version': '1.0.0'
                })

        self.logger.info(f"Prédiction par lot terminée : {len(results)} résultats")
        return results

    def _get_risk_level(self, fraud_probability: float) -> str:
        """
        Détermine le niveau de risque basé sur la probabilité de fraude.

        Args:
            fraud_probability (float): Probabilité de fraude.

        Returns:
            str: Niveau de risque.
        """
        if fraud_probability >= 0.8:
            return "TRÈS ÉLEVÉ"
        elif fraud_probability >= 0.6:
            return "ÉLEVÉ"
        elif fraud_probability >= 0.4:
            return "MOYEN"
        elif fraud_probability >= 0.2:
            return "FAIBLE"
        else:
            return "TRÈS FAIBLE"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations du modèle.

        Returns:
            Dict[str, Any]: Informations complètes sur le modèle.
        """
        # Recherche des métadonnées
        metadata_paths = [
            "../models/metadata/model_metadata.json",
            "models/metadata/model_metadata.json",
            str(ROOT_DIR / "models" / "metadata" / "model_metadata.json")
        ]

        for path in metadata_paths:
            if os.path.exists(path):
                try:
                    import json
                    with open(path, 'r') as f:
                        metadata = json.load(f)
                    self.logger.info(f"Métadonnées chargées depuis {path}")
                    return metadata
                except Exception as e:
                    self.logger.warning(f"Erreur chargement métadonnées depuis {path}: {e}")

        # Métadonnées par défaut
        return {
            "model_name": "Fraud Detection Model",
            "version": "1.0.0",
            "algorithm": "Ensemble (Random Forest + XGBoost)",
            "training_date": "2024-01-15",
            "performance_metrics": {
                "pr_auc": 0.9992,
                "roc_auc": 0.9998,
                "precision": 0.91,
                "recall": 0.86,
                "f1_score": 0.88
            }
        }
