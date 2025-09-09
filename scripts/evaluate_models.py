"""Script d'évaluation des modèles de détection de fraudes."""

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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve

# Imports locaux
from src.utils.helpers import setup_logging, load_config
from src.utils.metrics import calculate_metrics, pr_auc_score
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.feature_engineer import FeatureEngineer

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelEvaluator:
    """Classe pour l'évaluation complète des modèles de détection de fraudes."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise l'évaluateur de modèles.

        Args:
            config_path (str): Chemin vers le fichier de configuration
        """
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)

        # Chargement des données
        self._load_data()

        # Chargement du modèle
        self._load_model()

    def _load_data(self):
        """Charge et prépare les données de test."""
        self.logger.info("🔄 Chargement des données de test...")

        # Chargement des données
        data_loader = DataLoader()
        df = data_loader.load_data(file_path="data/raw/creditcard.csv")

        # Application du preprocessing et feature engineering
        preprocessor = Preprocessor(
            missing_strategy='median',
            scaling_strategy='standard'
        )
        df_prep = preprocessor.fit_transform(df)

        feature_engineer = FeatureEngineer()
        df_feat = feature_engineer.fit_transform(df_prep)

        # Séparation features/cible
        self.X = df_feat.drop('Class', axis=1)
        self.y = df_feat['Class']

        # Création d'un jeu de test représentatif
        from sklearn.model_selection import train_test_split
        _, self.X_test, _, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.logger.info(f"📊 Données de test : {self.X_test.shape}")
        self.logger.info(f"📈 Distribution : {self.y_test.value_counts(normalize=True).to_dict()}")

    def _load_model(self):
        """Charge le modèle entraîné."""
        import joblib

        # Recherche du modèle dans plusieurs emplacements possibles
        possible_paths = [
            "models/trained/best_model.joblib",
            "models/trained/best_model.pkl",
            "../models/trained/best_model.joblib",
            "../models/trained/best_model.pkl"
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
            "models/metadata/model_metadata.json",
            "../models/metadata/model_metadata.json"
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

    def evaluate_model(self):
        """Évalue le modèle sur le jeu de test."""
        self.logger.info("🔄 Évaluation du modèle...")

        # Prédictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_proba = self.model.predict_proba(self.X_test)

        # Gérer le cas où predict_proba retourne une seule colonne
        if self.y_proba.ndim == 1:
            self.y_proba = self.y_proba.reshape(-1, 1)
        elif self.y_proba.shape[1] == 1:
            # Si une seule classe, créer la probabilité complémentaire
            self.y_proba = np.column_stack([1 - self.y_proba.ravel(), self.y_proba.ravel()])

        # Métriques détaillées
        self.metrics = calculate_metrics(self.y_test, self.y_pred, self.y_proba)

        # Métriques supplémentaires
        self.additional_metrics = {
            'balanced_accuracy': (self.metrics['precision'] + self.metrics['recall']) / 2,
            'brier_score': brier_score_loss(self.y_test, self.y_proba[:, 1]),
            'average_precision': average_precision_score(self.y_test, self.y_proba[:, 1])
        }

        self.logger.info("✅ Évaluation terminée")
        return {**self.metrics, **self.additional_metrics}

    def generate_confusion_matrix(self):
        """Génère et sauvegarde la matrice de confusion."""
        self.logger.info("📊 Génération de la matrice de confusion...")

        cm = confusion_matrix(self.y_test, self.y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Légitime', 'Frauduleuse'],
                    yticklabels=['Légitime', 'Frauduleuse'])
        plt.title('Matrice de Confusion - Jeu de Test')
        plt.xlabel('Prédiction')
        plt.ylabel('Réalité')
        plt.tight_layout()

        # Sauvegarde
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig("reports/figures/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        return cm

    def generate_roc_curve(self):
        """Génère la courbe ROC."""
        self.logger.info("📊 Génération de la courbe ROC...")

        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label='.2f')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right")
        plt.grid(True)

        # Sauvegarde
        plt.savefig("reports/figures/roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

        return roc_auc

    def generate_pr_curve(self):
        """Génère la courbe Precision-Recall."""
        self.logger.info("📊 Génération de la courbe Precision-Recall...")

        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba[:, 1])
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label='.2f')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Rappel')
        plt.ylabel('Précision')
        plt.title('Courbe Precision-Recall')
        plt.legend(loc="lower left")
        plt.grid(True)

        # Sauvegarde
        plt.savefig("reports/figures/pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

        return pr_auc

    def analyze_thresholds(self):
        """Analyse les performances selon différents seuils."""
        self.logger.info("📊 Analyse des seuils de décision...")

        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_results = []

        for threshold in thresholds:
            y_pred_thresh = (self.y_proba >= threshold).astype(int)
            metrics_thresh = calculate_metrics(self.y_test, y_pred_thresh, self.y_proba.reshape(-1, 1))
            metrics_thresh['threshold'] = threshold
            threshold_results.append(metrics_thresh)

        # Sauvegarde des résultats
        threshold_df = pd.DataFrame(threshold_results)
        threshold_df.to_csv("reports/threshold_analysis.csv", index=False)

        # Visualisation
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(threshold_df['threshold'], threshold_df['precision'], 'b-', label='Precision')
        plt.plot(threshold_df['threshold'], threshold_df['recall'], 'r-', label='Recall')
        plt.xlabel('Seuil')
        plt.ylabel('Score')
        plt.title('Precision vs Recall')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(threshold_df['threshold'], threshold_df['f1'], 'g-', label='F1-Score')
        plt.xlabel('Seuil')
        plt.ylabel('F1-Score')
        plt.title('F1-Score vs Seuil')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(threshold_df['threshold'], threshold_df['precision'] * threshold_df['recall'],
                'purple', label='Precision × Recall')
        plt.xlabel('Seuil')
        plt.ylabel('Precision × Recall')
        plt.title('Precision × Recall')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.scatter(threshold_df['recall'], threshold_df['precision'],
                   c=threshold_df['threshold'], cmap='viridis', s=50)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Courbe Precision-Recall')
        plt.colorbar(label='Seuil')

        plt.tight_layout()
        plt.savefig("reports/figures/threshold_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        return threshold_df

    def generate_calibration_plot(self):
        """Génère le graphique de calibration."""
        self.logger.info("📊 Génération du graphique de calibration...")

        prob_true, prob_pred = calibration_curve(self.y_test, self.y_proba, n_bins=10)

        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', color='red', label='Modèle')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Calibration parfaite')
        plt.xlabel('Probabilité Prédite')
        plt.ylabel('Probabilité Observée')
        plt.title('Courbe de Calibration')
        plt.legend()
        plt.grid(True)

        # Sauvegarde
        plt.savefig("reports/figures/calibration_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        return prob_true, prob_pred

    def generate_report(self):
        """Génère un rapport d'évaluation complet."""
        self.logger.info("📋 Génération du rapport d'évaluation...")

        # Analyse des erreurs
        errors_df = pd.DataFrame({
            'y_true': self.y_test,
            'y_pred': self.y_pred,
            'y_proba': self.y_proba,
            'error': (self.y_test != self.y_pred).astype(int)
        })

        false_positives = len(errors_df[(errors_df['y_true'] == 0) & (errors_df['error'] == 1)])
        false_negatives = len(errors_df[(errors_df['y_true'] == 1) & (errors_df['error'] == 1)])

        # Rapport complet
        report = {
            'evaluation_info': {
                'date': datetime.now().isoformat(),
                'model_name': self.metadata.get('model_name', 'Unknown'),
                'test_size': len(self.X_test),
                'fraud_ratio': self.y_test.mean()
            },
            'metrics': {**self.metrics, **self.additional_metrics},
            'error_analysis': {
                'total_errors': errors_df['error'].sum(),
                'error_rate': errors_df['error'].mean(),
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'false_positive_rate': false_positives / (self.y_test == 0).sum(),
                'false_negative_rate': false_negatives / (self.y_test == 1).sum()
            },
            'threshold_analysis': {
                'optimal_threshold_f1': self.analyze_thresholds().loc[
                    self.analyze_thresholds()['f1'].idxmax(), 'threshold'
                ]
            }
        }

        # Sauvegarde du rapport
        os.makedirs("reports", exist_ok=True)
        with open("reports/evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Rapport texte simplifié
        text_report = f"""
RAPPORT D'ÉVALUATION DU MODÈLE
{'='*50}

Modèle évalué : {self.metadata.get('model_name', 'Unknown')}
Date d'évaluation : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Taille du test : {len(self.X_test)} échantillons

MÉTRIQUES PRINCIPALES :
{'-'*30}
PR-AUC : {self.metrics['pr_auc']:.4f}
ROC-AUC : {self.metrics['roc_auc']:.4f}
Precision : {self.metrics['precision']:.4f}
Recall : {self.metrics['recall']:.4f}
F1-Score : {self.metrics['f1']:.4f}
Balanced Accuracy : {self.additional_metrics['balanced_accuracy']:.4f}

ANALYSE DES ERREURS :
{'-'*30}
Taux d'erreur global : {errors_df['error'].mean()*100:.2f}%
Faux positifs : {false_positives}
Faux négatifs : {false_negatives}
Taux de faux positifs : {false_positives/(self.y_test==0).sum()*100:.2f}%
Taux de faux négatifs : {false_negatives/(self.y_test==1).sum()*100:.2f}%

RECOMMANDATIONS :
{'-'*30}
• Seuil optimal (F1) : {report['threshold_analysis']['optimal_threshold_f1']:.3f}
• Performance {'excellente' if self.metrics['pr_auc'] > 0.95 else 'bonne' if self.metrics['pr_auc'] > 0.85 else 'à améliorer'}
• {'Réduire les faux négatifs' if false_negatives > false_positives else 'Réduire les faux positifs'} en priorité

Rapport complet sauvegardé dans ../reports/evaluation_report.json
        """

        with open("reports/evaluation_summary.txt", 'w') as f:
            f.write(text_report)

        print(text_report)
        return report

    def run_full_evaluation(self):
        """Exécute l'évaluation complète."""
        self.logger.info("🚀 Démarrage de l'évaluation complète...")

        # Évaluation de base
        metrics = self.evaluate_model()

        # Génération des visualisations
        self.generate_confusion_matrix()
        self.generate_roc_curve()
        self.generate_pr_curve()
        self.generate_calibration_plot()

        # Analyses avancées
        self.analyze_thresholds()

        # Rapport final
        report = self.generate_report()

        self.logger.info("✅ Évaluation complète terminée")
        return report


def main():
    """Point d'entrée principal pour l'évaluation des modèles."""
    parser = argparse.ArgumentParser(description="Évaluation des modèles de détection de fraudes")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Chemin vers le fichier de configuration")
    parser.add_argument("--output", default="reports",
                       help="Dossier de sortie pour les rapports")
    parser.add_argument("--verbose", action="store_true",
                       help="Mode verbose")

    args = parser.parse_args()

    # Configuration du logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialisation de l'évaluateur
        evaluator = ModelEvaluator(args.config)

        # Exécution de l'évaluation complète
        report = evaluator.run_full_evaluation()

        print("\n🎉 ÉVALUATION TERMINÉE AVEC SUCCÈS !")
        print(f"📊 PR-AUC : {report['metrics']['pr_auc']:.4f}")
        print(f"📊 ROC-AUC : {report['metrics']['roc_auc']:.4f}")
        print(f"📋 Rapport complet : {args.output}/evaluation_report.json")

    except Exception as e:
        logging.error(f"❌ Erreur lors de l'évaluation : {str(e)}")
        raise


if __name__ == "__main__":
    main()