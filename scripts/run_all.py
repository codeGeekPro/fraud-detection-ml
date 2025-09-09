#!/usr/bin/env python3
"""
Script principal pour lancer tout le pipeline de détection de fraudes.

Ce script permet de :
- Vérifier l'environnement et les dépendances
- Lancer l'entraînement des modèles
- Évaluer les performances
- Démarrer l'API de prédiction
- Générer les rapports

Usage:
    python scripts/run_all.py --help
    python scripts/run_all.py --train --evaluate
    python scripts/run_all.py --api
    python scripts/run_all.py --all
"""

import sys
import os
import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/run_all.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """Classe principale pour gérer le pipeline de détection de fraudes."""

    def __init__(self):
        """Initialise le pipeline."""
        self.project_root = Path(__file__).resolve().parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.logs_dir = self.project_root / "logs"

        # Création des dossiers nécessaires
        self.logs_dir.mkdir(exist_ok=True)

        logger.info("🚀 Pipeline de détection de fraudes initialisé")

    def check_environment(self) -> bool:
        """
        Vérifie l'environnement et les dépendances.

        Returns:
            bool: True si l'environnement est prêt
        """
        logger.info("🔍 Vérification de l'environnement...")

        # Vérification Python
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error(f"❌ Python {python_version.major}.{python_version.minor} détecté. Python 3.8+ requis.")
            return False
        logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

        # Vérification des fichiers essentiels
        essential_files = [
            "config/config.yaml",
            "data/raw/creditcard.csv",
            "requirements.txt"
        ]

        for file_path in essential_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                logger.error(f"❌ Fichier manquant : {file_path}")
                return False
            logger.info(f"✅ {file_path}")

        # Vérification des dépendances
        try:
            import pandas as pd
            import numpy as np
            import sklearn
            import xgboost
            import fastapi
            logger.info("✅ Dépendances principales installées")
        except ImportError as e:
            logger.error(f"❌ Dépendance manquante : {e}")
            logger.info("💡 Installez avec : pip install -r requirements.txt")
            return False

        logger.info("🎉 Environnement prêt !")
        return True

    def run_training(self) -> bool:
        """
        Lance l'entraînement des modèles.

        Returns:
            bool: True si l'entraînement réussit
        """
        logger.info("🏋️ Lancement de l'entraînement...")

        try:
            # Changement vers le répertoire racine
            os.chdir(self.project_root)

            # Exécution du script d'entraînement
            cmd = [sys.executable, "scripts/train_models.py"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

            if result.returncode == 0:
                logger.info("✅ Entraînement terminé avec succès")
                logger.info("📊 Résultats de l'entraînement :")
                # Afficher les dernières lignes du output
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:  # Dernières 10 lignes
                    if line.strip():
                        logger.info(f"   {line}")
                return True
            else:
                logger.error("❌ Échec de l'entraînement")
                logger.error(f"Erreur : {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout de l'entraînement (30 minutes)")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement : {str(e)}")
            return False

    def run_evaluation(self) -> bool:
        """
        Lance l'évaluation des modèles.

        Returns:
            bool: True si l'évaluation réussit
        """
        logger.info("📊 Lancement de l'évaluation...")

        try:
            # Changement vers le répertoire racine
            os.chdir(self.project_root)

            # Exécution du script d'évaluation
            cmd = [sys.executable, "scripts/evaluate_models.py", "--verbose"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout

            if result.returncode == 0:
                logger.info("✅ Évaluation terminée avec succès")
                logger.info("📈 Résultats de l'évaluation :")
                # Extraire les métriques importantes
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if any(keyword in line.lower() for keyword in ['pr-auc', 'roc-auc', 'precision', 'recall', 'f1']):
                        logger.info(f"   {line.strip()}")
                return True
            else:
                logger.error("❌ Échec de l'évaluation")
                logger.error(f"Erreur : {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout de l'évaluation (10 minutes)")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'évaluation : {str(e)}")
            return False

    def run_api(self, host: str = "0.0.0.0", port: int = 8000) -> bool:
        """
        Lance l'API de prédiction.

        Args:
            host (str): Adresse d'écoute
            port (int): Port d'écoute

        Returns:
            bool: True si l'API démarre
        """
        logger.info(f"🌐 Lancement de l'API sur {host}:{port}...")

        try:
            # Changement vers le répertoire racine
            os.chdir(self.project_root)

            # Commande pour lancer l'API
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.api.app:app",
                "--host", host,
                "--port", str(port),
                "--reload"
            ]

            logger.info("🚀 API démarrée !")
            logger.info(f"📡 Accès : http://{host}:{port}")
            logger.info("📖 Documentation : http://{host}:{port}/docs")
            logger.info("💡 Appuyez sur Ctrl+C pour arrêter")

            # Lancement de l'API (bloquant)
            subprocess.run(cmd)

            return True

        except KeyboardInterrupt:
            logger.info("🛑 API arrêtée par l'utilisateur")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur lors du lancement de l'API : {str(e)}")
            return False

    def generate_report(self) -> bool:
        """
        Génère un rapport de synthèse.

        Returns:
            bool: True si le rapport est généré
        """
        logger.info("📋 Génération du rapport de synthèse...")

        try:
            # Vérification des fichiers de résultats
            model_path = self.project_root / "models" / "trained" / "best_model.joblib"
            report_path = self.project_root / "reports" / "evaluation_report.json"

            report_content = f"""
# Rapport de Synthèse - Pipeline de Détection de Fraudes

**Date :** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Statut :** {'✅ RÉUSSI' if model_path.exists() else '❌ ÉCHEC'}

## 📊 État du Système

### Fichiers Générés
- **Modèle :** {'✅ ' + str(model_path) if model_path.exists() else '❌ Manquant'}
- **Rapport d'évaluation :** {'✅ ' + str(report_path) if report_path.exists() else '❌ Manquant'}

### Métriques Clés
"""

            # Lecture du rapport d'évaluation si disponible
            if report_path.exists():
                import json
                with open(report_path, 'r') as f:
                    eval_data = json.load(f)

                metrics = eval_data.get('metrics', {})
                report_content += f"""
- **PR-AUC :** {metrics.get('pr_auc', 'N/A'):.4f}
- **ROC-AUC :** {metrics.get('roc_auc', 'N/A'):.4f}
- **Precision :** {metrics.get('precision', 'N/A'):.4f}
- **Recall :** {metrics.get('recall', 'N/A'):.4f}
- **F1-Score :** {metrics.get('f1', 'N/A'):.4f}
"""

            # Sauvegarde du rapport
            summary_path = self.project_root / "reports" / "pipeline_summary.md"
            summary_path.parent.mkdir(exist_ok=True)

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            logger.info(f"✅ Rapport généré : {summary_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération du rapport : {str(e)}")
            return False

    def run_pipeline(self, steps: List[str], api_host: str = "0.0.0.0", api_port: int = 8000) -> bool:
        """
        Exécute le pipeline complet.

        Args:
            steps (List[str]): Étapes à exécuter
            api_host (str): Host pour l'API
            api_port (int): Port pour l'API

        Returns:
            bool: True si toutes les étapes réussissent
        """
        logger.info("🎯 Démarrage du pipeline de détection de fraudes")
        logger.info(f"Étapes à exécuter : {', '.join(steps)}")

        success = True

        # Vérification de l'environnement
        if not self.check_environment():
            logger.error("❌ Environnement non prêt. Arrêt du pipeline.")
            return False

        # Entraînement
        if "train" in steps:
            if not self.run_training():
                success = False
                if "strict" in steps:
                    logger.error("❌ Arrêt du pipeline suite à l'échec de l'entraînement")
                    return False

        # Évaluation
        if "evaluate" in steps:
            if not self.run_evaluation():
                success = False
                if "strict" in steps:
                    logger.error("❌ Arrêt du pipeline suite à l'échec de l'évaluation")
                    return False

        # Rapport
        if "report" in steps:
            self.generate_report()

        # API
        if "api" in steps:
            self.run_api(api_host, api_port)

        if success:
            logger.info("🎉 Pipeline terminé avec succès !")
        else:
            logger.warning("⚠️ Pipeline terminé avec des avertissements")

        return success


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Pipeline complet de détection de fraudes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES D'UTILISATION :

1. Vérification de l'environnement :
   python scripts/run_all.py --check

2. Entraînement seulement :
   python scripts/run_all.py --train

3. Évaluation seulement :
   python scripts/run_all.py --evaluate

4. Pipeline complet (entraînement + évaluation) :
   python scripts/run_all.py --all

5. Pipeline complet + API :
   python scripts/run_all.py --all --api

6. API seulement :
   python scripts/run_all.py --api --host 127.0.0.1 --port 8080

7. Mode strict (arrêt en cas d'erreur) :
   python scripts/run_all.py --all --strict
        """
    )

    # Options principales
    parser.add_argument("--check", action="store_true",
                       help="Vérifier l'environnement seulement")
    parser.add_argument("--train", action="store_true",
                       help="Lancer l'entraînement des modèles")
    parser.add_argument("--evaluate", action="store_true",
                       help="Lancer l'évaluation des modèles")
    parser.add_argument("--report", action="store_true",
                       help="Générer le rapport de synthèse")
    parser.add_argument("--api", action="store_true",
                       help="Lancer l'API de prédiction")

    # Options combinées
    parser.add_argument("--all", action="store_true",
                       help="Exécuter tout le pipeline (train + evaluate + report)")

    # Options API
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host pour l'API (défaut: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port pour l'API (défaut: 8000)")

    # Options générales
    parser.add_argument("--strict", action="store_true",
                       help="Arrêter le pipeline en cas d'erreur")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Mode verbose")

    args = parser.parse_args()

    # Configuration du logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialisation du pipeline
    pipeline = FraudDetectionPipeline()

    # Détermination des étapes à exécuter
    steps = []
    if args.check:
        steps.append("check")
    if args.train or args.all:
        steps.append("train")
    if args.evaluate or args.all:
        steps.append("evaluate")
    if args.report or args.all:
        steps.append("report")
    if args.api:
        steps.append("api")
    if args.strict:
        steps.append("strict")

    # Si aucune option spécifiée, afficher l'aide
    if not steps:
        parser.print_help()
        return

    # Exécution du pipeline
    try:
        success = pipeline.run_pipeline(steps, args.host, args.port)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur inattendue : {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
