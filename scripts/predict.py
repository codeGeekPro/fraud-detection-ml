"""Script de prédiction en ligne de commande."""

import sys
import os
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ajout du chemin racine au sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Imports de base
import numpy as np
import pandas as pd

# Imports locaux
from src.utils.helpers import setup_logging, load_config
from src.data.preprocessor import Preprocessor
from src.data.feature_engineer import FeatureEngineer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FraudPredictor:
    """Classe pour effectuer des prédictions de fraudes."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise le prédicteur de fraudes.

        Args:
            config_path (str): Chemin vers le fichier de configuration
        """
        self.config = load_config(config_path)
        self.logger = logger

        # Chargement du modèle
        self._load_model()

        # Chargement des composants de preprocessing
        self._load_preprocessing_components()

    def _load_model(self):
        """Charge le modèle entraîné."""
        import joblib

        # Recherche du modèle dans plusieurs emplacements possibles
        possible_paths = [
            "../models/trained/best_model.joblib",
            "../models/trained/best_model.pkl",
            "models/trained/best_model.joblib",
            "models/trained/best_model.pkl"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            self.model = joblib.load(model_path)
            self.logger.info(f"✅ Modèle chargé avec succès depuis {model_path}")
        else:
            raise FileNotFoundError(f"Modèle non trouvé. Chemins essayés : {possible_paths}")

        # Chargement des métadonnées
        metadata_paths = [
            "../models/metadata/model_metadata.json",
            "models/metadata/model_metadata.json"
        ]
        
        metadata_path = None
        for path in metadata_paths:
            if os.path.exists(path):
                metadata_path = path
                break
        
        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.logger.info("✅ Métadonnées chargées")
        else:
            self.metadata = {}
            self.logger.warning("⚠️ Métadonnées non trouvées")

    def _load_preprocessing_components(self):
        """Charge les composants de preprocessing."""
        # Chargement des données d'entraînement pour fitter les transformeurs
        from src.data.data_loader import DataLoader

        data_loader = DataLoader()
        df = data_loader.load_data(file_path="../data/raw/creditcard.csv")

        # Initialisation du preprocessor
        self.preprocessor = Preprocessor({
            'missing_values_strategy': 'median',
            'scaling_method': 'standard'
        })

        # Fit du preprocessor (exclut automatiquement la colonne Class)
        self.preprocessor.fit(df)

        # Initialisation du feature engineer
        self.feature_engineer = FeatureEngineer()

        # Fit du feature engineer
        df_prep = self.preprocessor.transform(df)
        self.feature_engineer.fit(df_prep)

        self.logger.info("✅ Composants de preprocessing chargés")

    def _validate_input(self, transaction_data: dict) -> pd.DataFrame:
        """
        Valide et prépare les données d'entrée.

        Args:
            transaction_data (dict): Données de la transaction

        Returns:
            pd.DataFrame: Données validées et préparées
        """
        required_features = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ]

        # Vérification des features requises
        missing_features = [f for f in required_features if f not in transaction_data]
        if missing_features:
            raise ValueError(f"Features manquantes : {missing_features}")

        # Conversion en DataFrame
        df = pd.DataFrame([transaction_data])

        # Validation des types de données
        numeric_features = required_features
        for feature in numeric_features:
            try:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
            except:
                raise ValueError(f"Feature {feature} doit être numérique")

        # Vérification des valeurs manquantes
        if df.isnull().any().any():
            raise ValueError("Les données contiennent des valeurs manquantes")

        return df

    def predict_single(self, transaction_data: dict, threshold: float = 0.5) -> dict:
        """
        Effectue une prédiction pour une seule transaction.

        Args:
            transaction_data (dict): Données de la transaction
            threshold (float): Seuil de décision

        Returns:
            dict: Résultats de la prédiction
        """
        self.logger.info("🔄 Prédiction en cours...")

        # Validation et préparation des données
        df_input = self._validate_input(transaction_data)

        # Application du preprocessing
        df_prep = self.preprocessor.transform(df_input)

        # Application du feature engineering
        df_features = self.feature_engineer.transform(df_prep)

        # Suppression de la colonne Class si elle existe
        if 'Class' in df_features.columns:
            df_features = df_features.drop('Class', axis=1)

        # Prédiction
        prediction_proba = self.model.predict_proba(df_features)[0]
        prediction = (prediction_proba[1] >= threshold).astype(int)

        # Résultats détaillés
        result = {
            'prediction': int(prediction),
            'prediction_proba': {
                'fraud': float(prediction_proba[1]),
                'legitimate': float(prediction_proba[0])
            },
            'threshold_used': threshold,
            'is_fraud': bool(prediction),
            'confidence': float(max(prediction_proba)),
            'risk_level': self._get_risk_level(prediction_proba[1]),
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'name': self.metadata.get('model_name', 'Unknown'),
                'version': self.metadata.get('version', 'Unknown')
            }
        }

        self.logger.info(f"✅ Prédiction terminée : {'FRAUDE' if prediction else 'LÉGITIME'} "
                        f"(probabilité : {prediction_proba[1]:.4f})")

        return result

    def _get_risk_level(self, fraud_probability: float) -> str:
        """
        Détermine le niveau de risque basé sur la probabilité de fraude.

        Args:
            fraud_probability (float): Probabilité de fraude

        Returns:
            str: Niveau de risque
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

    def predict_batch(self, transactions_data: list, threshold: float = 0.5) -> list:
        """
        Effectue des prédictions pour un lot de transactions.

        Args:
            transactions_data (list): Liste des données de transactions
            threshold (float): Seuil de décision

        Returns:
            list: Liste des résultats de prédiction
        """
        self.logger.info(f"🔄 Prédiction par lot : {len(transactions_data)} transactions")

        results = []
        for i, transaction in enumerate(transactions_data):
            try:
                result = self.predict_single(transaction, threshold)
                result['transaction_id'] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Erreur pour la transaction {i} : {str(e)}")
                results.append({
                    'transaction_id': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        self.logger.info(f"✅ Prédiction par lot terminée : {len(results)} résultats")
        return results

    def save_prediction(self, result: dict, output_path: str = None):
        """
        Sauvegarde le résultat de prédiction.

        Args:
            result (dict): Résultat de la prédiction
            output_path (str): Chemin de sauvegarde
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"../reports/prediction_{timestamp}.json"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        self.logger.info(f"💾 Prédiction sauvegardée : {output_path}")


def create_sample_transaction() -> dict:
    """Crée un exemple de transaction pour les tests."""
    return {
        "Time": 123456.0,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }


def main():
    """Point d'entrée principal pour les prédictions."""
    parser = argparse.ArgumentParser(
        description="Prédiction de fraudes pour transactions individuelles ou par lot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES D'UTILISATION :

1. Prédiction pour une transaction depuis un fichier JSON :
   python predict.py --input transaction.json

2. Prédiction pour une transaction depuis la ligne de commande :
   python predict.py --features "Time=123456,V1=-1.35,V2=-0.07,Amount=149.62"

3. Prédiction par lot :
   python predict.py --batch transactions.json

4. Utilisation d'un seuil personnalisé :
   python predict.py --input transaction.json --threshold 0.3

5. Génération d'un exemple de transaction :
   python predict.py --sample
        """
    )

    parser.add_argument("--config", default="../config/config.yaml",
                       help="Chemin vers le fichier de configuration")
    parser.add_argument("--input", "-i", type=str,
                       help="Fichier JSON contenant les données de transaction")
    parser.add_argument("--batch", "-b", type=str,
                       help="Fichier JSON contenant un lot de transactions")
    parser.add_argument("--features", "-f", type=str,
                       help="Features sous forme clé=valeur séparées par des virgules")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Seuil de décision (défaut: 0.5)")
    parser.add_argument("--output", "-o", type=str,
                       help="Fichier de sortie pour les résultats")
    parser.add_argument("--sample", action="store_true",
                       help="Génère un exemple de transaction")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Mode verbose")

    args = parser.parse_args()

    # Configuration du logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Génération d'exemple
        if args.sample:
            sample = create_sample_transaction()
            print("EXEMPLE DE TRANSACTION :")
            print(json.dumps(sample, indent=2))
            return

        # Initialisation du prédicteur
        predictor = FraudPredictor(args.config)

        # Prédiction depuis fichier
        if args.input:
            with open(args.input, 'r') as f:
                transaction_data = json.load(f)
            result = predictor.predict_single(transaction_data, args.threshold)

        # Prédiction par lot
        elif args.batch:
            with open(args.batch, 'r') as f:
                transactions_data = json.load(f)
            result = predictor.predict_batch(transactions_data, args.threshold)

        # Prédiction depuis ligne de commande
        elif args.features:
            # Parsing des features
            features_dict = {}
            for pair in args.features.split(','):
                key, value = pair.split('=')
                features_dict[key.strip()] = float(value.strip())
            result = predictor.predict_single(features_dict, args.threshold)

        else:
            parser.print_help()
            return

        # Affichage des résultats
        if isinstance(result, list):
            # Résultats par lot
            print(f"\n📊 RÉSULTATS DE PRÉDICTION PAR LOT ({len(result)} transactions) :\n")
            fraud_count = sum(1 for r in result if r.get('is_fraud', False))
            print(f"🔍 Transactions frauduleuses détectées : {fraud_count}")
            print(f"✅ Transactions légitimes : {len(result) - fraud_count}")

            # Affichage détaillé des fraudes
            if fraud_count > 0:
                print("\n🚨 TRANSACTIONS SUSPECTES :")
                for r in result:
                    if r.get('is_fraud', False):
                        print(f"  • Transaction {r['transaction_id']}: "
                              f"Probabilité = {r['prediction_proba']['fraud']:.4f}, "
                              f"Risque = {r['risk_level']}")

        else:
            # Résultat unique
            print("\n📊 RÉSULTAT DE PRÉDICTION :")
            print(f"🔍 Prédiction : {'🚨 FRAUDE' if result['is_fraud'] else '✅ LÉGITIME'}")
            print(f"📈 Probabilité de fraude : {result['prediction_proba']['fraud']:.4f}")
            print(f"🎯 Confiance : {result['confidence']:.4f}")
            print(f"⚠️ Niveau de risque : {result['risk_level']}")
            print(f"📏 Seuil utilisé : {result['threshold_used']}")

        # Sauvegarde si demandé
        if args.output:
            predictor.save_prediction(result, args.output)

        print("\n🎉 PRÉDICTION TERMINÉE AVEC SUCCÈS !")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction : {str(e)}")
        raise


if __name__ == "__main__":
    main()
