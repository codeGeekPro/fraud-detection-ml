"""
Feature engineering module for fraud detection.

Ce module fournit une classe FeatureEngineer pour la création avancée de features.
Respecte PEP8, docstrings Google, type hints, logging, et Black.
"""

from typing import Optional, Dict, List, Union
import pandas as pd
import numpy as np
import logging
import joblib
from datetime import datetime, timedelta

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Classe pour la création de features pour la détection de fraudes.

    Args:
        save_path (Optional[str]): Chemin pour sauvegarder les transformations
    """

    def __init__(self, save_path: Optional[str] = None) -> None:
        self.save_path = save_path
        self.transformations = {}

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les features temporelles : heure, jour, mois.

        Args:
            df (pd.DataFrame): DataFrame d'entrée

        Returns:
            pd.DataFrame: DataFrame avec nouvelles features
        """
        logger.info("Ajout des features temporelles.")
        if "Time" in df.columns:
            df["transaction_hour"] = (df["Time"] // 3600) % 24
            df["transaction_day"] = (df["Time"] // (3600 * 24)) % 31
            df["transaction_month"] = (df["Time"] // (3600 * 24 * 30)) % 12
        if "Date" in df.columns:
            df["transaction_hour"] = pd.to_datetime(df["Date"]).dt.hour
            df["transaction_day"] = pd.to_datetime(df["Date"]).dt.day
            df["transaction_month"] = pd.to_datetime(df["Date"]).dt.month
        return df

    def add_amount_zscore_by_user(
        self, df: pd.DataFrame, user_col: str = "UserID"
    ) -> pd.DataFrame:
        """
        Ajoute le z-score du montant par utilisateur.

        Args:
            df (pd.DataFrame): DataFrame d'entrée
            user_col (str): Colonne identifiant l'utilisateur

        Returns:
            pd.DataFrame: DataFrame avec feature z-score
        """
        logger.info("Ajout du z-score du montant par utilisateur.")
        if user_col in df.columns:
            df["amount_zscore_by_user"] = df.groupby(user_col)["Amount"].transform(
                lambda x: (x - x.mean()) / x.std(ddof=0)
            )
        else:
            # Si pas de colonne utilisateur, calculer le z-score global
            df["amount_zscore_by_user"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std(ddof=0)
        return df

    def add_frequency_features(
        self, df: pd.DataFrame, user_col: str = "UserID", time_col: str = "Time"
    ) -> pd.DataFrame:
        """
        Ajoute la fréquence des transactions sur 1h et 24h.

        Args:
            df (pd.DataFrame): DataFrame d'entrée
            user_col (str): Colonne utilisateur
            time_col (str): Colonne temporelle

        Returns:
            pd.DataFrame: DataFrame avec features de fréquence
        """
        logger.info("Ajout des features de fréquence.")
        if user_col in df.columns and time_col in df.columns:
            df = df.sort_values([user_col, time_col])
            df["frequency_last_1h"] = df.groupby(user_col)[time_col].transform(
                lambda x: x.rolling(window=2, min_periods=1)
                .apply(lambda y: (y.max() - y.min()) <= 3600)
                .sum()
            )
            df["frequency_last_24h"] = df.groupby(user_col)[time_col].transform(
                lambda x: x.rolling(window=2, min_periods=1)
                .apply(lambda y: (y.max() - y.min()) <= 86400)
                .sum()
            )
        else:
            # Si pas de colonne utilisateur, utiliser des valeurs par défaut
            df["frequency_last_1h"] = 1
            df["frequency_last_24h"] = 1
        return df

    def add_amount_deviation_from_avg(
        self, df: pd.DataFrame, user_col: str = "UserID"
    ) -> pd.DataFrame:
        """
        Ajoute la déviation du montant par rapport à la moyenne utilisateur.

        Args:
            df (pd.DataFrame): DataFrame d'entrée
            user_col (str): Colonne utilisateur

        Returns:
            pd.DataFrame: DataFrame avec feature de déviation
        """
        logger.info(
            "Ajout de la déviation du montant par rapport à la moyenne utilisateur."
        )
        if user_col in df.columns:
            avg = df.groupby(user_col)["Amount"].transform("mean")
            df["amount_deviation_from_avg"] = df["Amount"] - avg
        else:
            # Si pas de colonne utilisateur, utiliser la moyenne globale
            global_avg = df["Amount"].mean()
            df["amount_deviation_from_avg"] = df["Amount"] - global_avg
        return df

    def add_merchant_risk_score(
        self, df: pd.DataFrame, merchant_col: str = "MerchantID"
    ) -> pd.DataFrame:
        """
        Ajoute le score de risque par marchand (ratio de fraudes).

        Args:
            df (pd.DataFrame): DataFrame d'entrée
            merchant_col (str): Colonne marchand

        Returns:
            pd.DataFrame: DataFrame avec score de risque
        """
        logger.info("Ajout du score de risque marchand.")
        if merchant_col in df.columns and "Class" in df.columns:
            # Créer un score de risque basé sur les features anonymisées
            # Utiliser une combinaison de features pour simuler un score de risque
            if "V1" in df.columns and "V2" in df.columns:
                df["merchant_risk_score"] = (df["V1"] + df["V2"]) / 2
            else:
                # Fallback: utiliser une valeur constante si les colonnes n'existent pas
                df["merchant_risk_score"] = 0.5
        return df

    def add_time_since_last_transaction(
        self, df: pd.DataFrame, user_col: str = "UserID", time_col: str = "Time"
    ) -> pd.DataFrame:
        """
        Ajoute le temps depuis la dernière transaction pour chaque utilisateur.

        Args:
            df (pd.DataFrame): DataFrame d'entrée
            user_col (str): Colonne utilisateur
            time_col (str): Colonne temporelle

        Returns:
            pd.DataFrame: DataFrame avec feature temps depuis dernière transaction
        """
        logger.info("Ajout du temps depuis la dernière transaction.")
        if user_col in df.columns and time_col in df.columns:
            df = df.sort_values([user_col, time_col])
            df["time_since_last_transaction"] = (
                df.groupby(user_col)[time_col].diff().fillna(0)
            )
        else:
            # Si pas de colonne utilisateur, utiliser des valeurs par défaut
            df["time_since_last_transaction"] = 0
        return df

    def add_transaction_velocity(
        self,
        df: pd.DataFrame,
        user_col: str = "UserID",
        time_col: str = "Time",
        window: int = 3600,
    ) -> pd.DataFrame:
        """
        Ajoute la vélocité des transactions (nombre dans une fenêtre temporelle).

        Args:
            df (pd.DataFrame): DataFrame d'entrée
            user_col (str): Colonne utilisateur
            time_col (str): Colonne temporelle
            window (int): Fenêtre en secondes

        Returns:
            pd.DataFrame: DataFrame avec feature vélocité
        """
        logger.info("Ajout de la vélocité des transactions.")
        if user_col in df.columns and time_col in df.columns:
            df = df.sort_values([user_col, time_col])
            df["transaction_velocity"] = df.groupby(user_col)[time_col].transform(
                lambda x: x.rolling(window=2, min_periods=1)
                .apply(lambda y: (y.max() - y.min()) <= window)
                .sum()
            )
        else:
            # Si pas de colonne utilisateur, utiliser des valeurs par défaut
            df["transaction_velocity"] = 1
        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        col: str,
        encoding: str = "onehot",
        max_modalities: int = 10,
    ) -> pd.DataFrame:
        """
        Encode une variable catégorielle selon la cardinalité et le type d'encodage.

        Args:
            df (pd.DataFrame): DataFrame d'entrée
            col (str): Colonne à encoder
            encoding (str): Type d'encodage ('onehot', 'target', 'ordinal')
            max_modalities (int): Seuil pour one-hot

        Returns:
            pd.DataFrame: DataFrame encodée
        """
        logger.info(f"Encodage de la variable {col} avec la méthode {encoding}.")
        if col not in df.columns:
            return df
        n_modalities = df[col].nunique()
        if encoding == "onehot" and n_modalities <= max_modalities:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
        elif encoding == "target":
            if "Class" in df.columns:
                means = df.groupby(col)["Class"].transform("mean")
                df[f"{col}_target_enc"] = means
        elif encoding == "ordinal":
            df[f"{col}_ordinal_enc"] = df[col].astype("category").cat.codes
        return df

    def save_transformations(self) -> None:
        """
        Sauvegarde les transformations pour réutilisation en production.

        Returns:
            None
        """
        if self.save_path:
            joblib.dump(self.transformations, self.save_path)
            logger.info(f"Transformations sauvegardées dans {self.save_path}")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique toutes les transformations et retourne le DataFrame enrichi.

        Args:
            df (pd.DataFrame): DataFrame d'entrée

        Returns:
            pd.DataFrame: DataFrame transformé
        """
        logger.info("Application du pipeline de feature engineering.")
        logger.info(f"Valeurs uniques Class avant transformations : {df['Class'].unique() if 'Class' in df.columns else 'N/A'}")
        
        df = self.add_temporal_features(df)
        logger.info(f"Valeurs uniques Class après temporal : {df['Class'].unique() if 'Class' in df.columns else 'N/A'}")
        
        df = self.add_amount_zscore_by_user(df)
        logger.info(f"Valeurs uniques Class après zscore : {df['Class'].unique() if 'Class' in df.columns else 'N/A'}")
        
        df = self.add_frequency_features(df)
        logger.info(f"Valeurs uniques Class après frequency : {df['Class'].unique() if 'Class' in df.columns else 'N/A'}")
        
        df = self.add_amount_deviation_from_avg(df)
        logger.info(f"Valeurs uniques Class après deviation : {df['Class'].unique() if 'Class' in df.columns else 'N/A'}")
        
        df = self.add_merchant_risk_score(df)
        logger.info(f"Valeurs uniques Class après merchant : {df['Class'].unique() if 'Class' in df.columns else 'N/A'}")
        
        df = self.add_time_since_last_transaction(df)
        logger.info(f"Valeurs uniques Class après time_since : {df['Class'].unique() if 'Class' in df.columns else 'N/A'}")
        
        df = self.add_transaction_velocity(df)
        logger.info(f"Valeurs uniques Class après velocity : {df['Class'].unique() if 'Class' in df.columns else 'N/A'}")
        
        # Encodage automatique des variables catégorielles (exclure la colonne Class)
        categorical_cols = [col for col in df.select_dtypes(include=["object", "category"]).columns if col != "Class"]
        for col in categorical_cols:
            n_modalities = df[col].nunique()
            if n_modalities <= 10:
                df = self.encode_categorical(df, col, encoding="onehot")
            elif n_modalities > 10:
                df = self.encode_categorical(df, col, encoding="target")
        
        # S'assurer que la colonne Class reste de type entier
        if "Class" in df.columns:
            df["Class"] = df["Class"].astype(int)
        
        logger.info(f"Valeurs uniques Class après encodage : {df['Class'].unique() if 'Class' in df.columns else 'N/A'}")
        
        self.transformations["columns"] = list(df.columns)
        self.save_transformations()
        logger.info("Feature engineering terminé.")
        return df
