"""
Module de chargement des données pour la détection de fraudes.

Ce module fournit des fonctionnalités pour charger et valider les données
de transactions bancaires depuis différentes sources (CSV, Parquet).
"""

from typing import Optional, Union, Dict, Tuple
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Classe pour charger et valider les données de transactions bancaires.

    Cette classe gère le chargement des données depuis différents formats
    et effectue des validations de base sur les données chargées.

    Attributes:
        config_path (str): Chemin vers le fichier de configuration
        data_config (dict): Configuration pour le chargement des données
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise le DataLoader avec la configuration spécifiée.

        Args:
            config_path (str): Chemin vers le fichier de configuration YAML
        """
        # Convertir en chemin absolu relatif au répertoire racine du projet
        if not Path(config_path).is_absolute():
            # Trouver le répertoire racine du projet (celui qui contient src/)
            current_path = Path(__file__).resolve()
            project_root = current_path.parent.parent.parent  # Remonter de 3 niveaux: src/data/data_loader.py -> src -> projet
            self.config_path = str(project_root / config_path)
        else:
            self.config_path = config_path
            
        self.data_config = self._load_config()

    def _load_config(self) -> Dict:
        """
        Charge la configuration depuis le fichier YAML si applicable.

        Returns:
            Dict: Configuration chargée

        Raises:
            FileNotFoundError: Si le fichier de configuration n'existe pas
        """
        if self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
            try:
                with open(self.config_path, "r") as file:
                    config = yaml.safe_load(file)
                    return config["data"]
            except FileNotFoundError:
                logger.error(f"Fichier de configuration non trouvé: {self.config_path}")
                raise
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
                raise
        else:
            logger.warning("Le fichier spécifié n'est pas un fichier YAML. Ignoré.")
            return {}

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV ou Parquet.

        Args:
            file_path (str, optional): Chemin vers le fichier de données.
                Si non spécifié, utilise le chemin dans la configuration.

        Returns:
            pd.DataFrame: DataFrame contenant les données chargées

        Raises:
            FileNotFoundError: Si le fichier de données n'existe pas
            ValueError: Si le format de fichier n'est pas supporté
        """
        if file_path is None:
            if "raw_data_path" in self.data_config:
                file_path = self.data_config["raw_data_path"]
            else:
                raise KeyError("Le chemin du fichier de données n'est pas spécifié et la configuration est absente.")

        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"Fichier non trouvé: {file_path}")
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")

        try:
            if file_path.suffix.lower() == ".csv":
                data = pd.read_csv(file_path)
                logger.info(f"Données CSV chargées depuis {file_path}")
            elif file_path.suffix.lower() == ".parquet":
                data = pd.read_parquet(file_path)
                logger.info(f"Données Parquet chargées depuis {file_path}")
            else:
                raise ValueError(f"Format de fichier non supporté: {file_path.suffix}")

            return self._validate_data(data)

        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données en supprimant ou corrigeant les valeurs invalides.

        Args:
            data (pd.DataFrame): Données brutes à nettoyer

        Returns:
            pd.DataFrame: Données nettoyées
        """
        # Filtrer les valeurs invalides dans la colonne 'Class'
        valid_classes = {0, 1}
        if not set(data['Class'].unique()).issubset(valid_classes):
            logger.warning("Des valeurs invalides ont été détectées dans la colonne 'Class'. Elles seront supprimées.")
            data = data[data['Class'].isin(valid_classes)]

        return data

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Valide le DataFrame chargé et effectue des vérifications de base.

        Args:
            data (pd.DataFrame): DataFrame à valider

        Returns:
            pd.DataFrame: DataFrame validé

        Raises:
            ValueError: Si les données ne respectent pas les critères de validation
        """
        # Nettoyage des données avant validation
        data = self._clean_data(data)

        # Vérification des colonnes requises
        required_columns = ["Time", "Amount", "Class"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes: {missing_columns}")

        # Vérification des types de données
        if not pd.api.types.is_numeric_dtype(data["Time"]):
            raise ValueError("La colonne 'Time' doit être numérique")
        if not pd.api.types.is_numeric_dtype(data["Amount"]):
            raise ValueError("La colonne 'Amount' doit être numérique")
        if not pd.api.types.is_numeric_dtype(data["Class"]):
            raise ValueError("La colonne 'Class' doit être numérique")

        # Vérification des valeurs manquantes
        if data.isnull().any().any():
            logger.warning("Des valeurs manquantes ont été détectées dans les données")

        # Vérification des montants négatifs
        if (data["Amount"] < 0).any():
            logger.warning("Des montants négatifs ont été détectés")

        return data

    def split_data(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divise les données en ensembles d'entraînement, validation et test.

        Args:
            data (pd.DataFrame): DataFrame à diviser

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                (train_data, val_data, test_data)
        """
        # Chargement des paramètres de split depuis la configuration
        test_size = self.data_config.get("test_size", 0.15)
        val_size = self.data_config.get("val_size", 0.15)
        random_state = self.data_config.get("random_state", 42)

        # Calcul des indices pour la stratification
        from sklearn.model_selection import train_test_split

        # Premier split pour séparer les données de test
        train_val_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state, stratify=data["Class"]
        )

        # Second split pour séparer les données d'entraînement et de validation
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=train_val_data["Class"],
        )

        # Sauvegarde des données
        self._save_splits(train_data, val_data, test_data)

        return train_data, val_data, test_data

    def _save_splits(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> None:
        """
        Sauvegarde les différents splits de données au format Parquet.

        Args:
            train_data (pd.DataFrame): Données d'entraînement
            val_data (pd.DataFrame): Données de validation
            test_data (pd.DataFrame): Données de test
        """
        try:
            # Création des dossiers si nécessaire
            processed_dir = Path(self.data_config["processed_data_path"]).parent
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Sauvegarde des données
            train_data.to_parquet(self.data_config["train_data_path"])
            val_data.to_parquet(self.data_config["val_data_path"])
            test_data.to_parquet(self.data_config["test_data_path"])

            logger.info("Données sauvegardées avec succès dans le dossier processed/")

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données: {str(e)}")
            raise


if __name__ == "__main__":
    # Exemple d'utilisation
    loader = DataLoader("config/config.yaml")
    data = loader.load_data()
    train_data, val_data, test_data = loader.split_data(data)

    print(f"Dimensions des données :")
    print(f"Train : {train_data.shape}")
    print(f"Validation : {val_data.shape}")
    print(f"Test : {test_data.shape}")
