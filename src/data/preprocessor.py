"""
Preprocessor module for fraud detection.

Ce module fournit une classe Preprocessor compatible sklearn pour le pipeline de prétraitement.
Respecte PEP8, docstrings Google, type hints, logging, et Black.
"""

from typing import Optional, Dict, Union, List
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Classe de prétraitement des données pour la détection de fraudes.

    Cette classe gère l'imputation des valeurs manquantes, la normalisation/standardisation,
    et la création de pipelines réutilisables.

    Args:
        missing_strategy (str): Stratégie d'imputation ('mean', 'median', 'mode', 'knn')
        scaling_strategy (str): Stratégie de scaling ('standard', 'minmax', 'robust', None)
        knn_neighbors (int): Nombre de voisins pour KNNImputer
    """

    def __init__(
        self,
        missing_strategy: str = "median",
        scaling_strategy: Optional[str] = "standard",
        knn_neighbors: int = 5,
    ) -> None:
        self.missing_strategy = missing_strategy
        self.scaling_strategy = scaling_strategy
        self.knn_neighbors = knn_neighbors
        self.imputer = None
        self.scaler = None
        self.pipeline = None
        self.numeric_cols = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Preprocessor":
        """
        Apprend les paramètres d'imputation et de scaling sur les données d'entraînement.

        Args:
            X (pd.DataFrame): Données d'entrée
            y (Optional[pd.Series]): Labels (non utilisé)

        Returns:
            Preprocessor: Instance fitted
        """
        logger.info(
            f"Initialisation du pipeline de préprocessing: missing_strategy={self.missing_strategy}, scaling_strategy={self.scaling_strategy}"
        )
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        steps: List = []

        # Imputation
        if self.missing_strategy == "mean":
            self.imputer = SimpleImputer(strategy="mean")
        elif self.missing_strategy == "median":
            self.imputer = SimpleImputer(strategy="median")
        elif self.missing_strategy == "mode":
            self.imputer = SimpleImputer(strategy="most_frequent")
        elif self.missing_strategy == "knn":
            self.imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        else:
            raise ValueError(
                f"Stratégie d'imputation non supportée: {self.missing_strategy}"
            )
        steps.append(("imputer", self.imputer))

        # Scaling
        if self.scaling_strategy == "standard":
            self.scaler = StandardScaler()
            steps.append(("scaler", self.scaler))
        elif self.scaling_strategy == "minmax":
            self.scaler = MinMaxScaler()
            steps.append(("scaler", self.scaler))
        elif self.scaling_strategy == "robust":
            self.scaler = RobustScaler()
            steps.append(("scaler", self.scaler))
        elif self.scaling_strategy is None:
            self.scaler = None
        else:
            raise ValueError(
                f"Stratégie de scaling non supportée: {self.scaling_strategy}"
            )

        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X[self.numeric_cols])
        logger.info("Pipeline de prétraitement entraîné sur colonnes numériques.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme les données en appliquant l'imputation et le scaling uniquement sur les colonnes numériques.

        Args:
            X (pd.DataFrame): Données à transformer

        Returns:
            pd.DataFrame: Données transformées
        """
        logger.info("Transformation des données avec le pipeline de prétraitement.")
        X_num = X[self.numeric_cols]
        X_num_trans = self.pipeline.transform(X_num)
        X_out = X.copy()
        X_out[self.numeric_cols] = X_num_trans
        return X_out

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit et transforme les données d'entrée.

        Args:
            X (pd.DataFrame): Données à transformer
            y (Optional[pd.Series]): Labels (non utilisé)

        Returns:
            pd.DataFrame: Données transformées
        """
        self.fit(X, y)
        return self.transform(X)

    def get_pipeline(self) -> Pipeline:
        """
        Retourne le pipeline sklearn construit.

        Returns:
            Pipeline: Pipeline sklearn
        """
        return self.pipeline

    def get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """
        Retourne la liste des noms de features après transformation.

        Args:
            X (pd.DataFrame): Données d'entrée

        Returns:
            List[str]: Noms des features
        """
        return list(X.columns)
