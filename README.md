# 🔍 Détection de Fraudes Bancaires avec Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen.svg)](https://github.com/username/fraud-detection-ml)

## 📋 Description

Système de détection de fraudes bancaires basé sur le Machine Learning, utilisant des algorithmes avancés (Random Forest et XGBoost) pour identifier automatiquement les transactions frauduleuses. Le projet met l'accent sur la gestion efficace des données déséquilibrées et l'interprétabilité des résultats.

Le système intègre des techniques avancées de feature engineering temporel, une gestion robuste des valeurs aberrantes, et une optimisation automatique des hyperparamètres. Le pipeline de données est entièrement automatisé, de l'ingestion à la prédiction.

### 🎯 Caractéristiques Principales

- **Preprocessing Avancé**
  - Normalisation robuste des features numériques
  - Gestion intelligente des valeurs manquantes
  - Exclusion automatique des colonnes sensibles du preprocessing

- **Feature Engineering Sophistiqué**
  - Features temporelles (heure, jour, mois)
  - Z-score et déviations par montant
  - Analyse de fréquence des transactions
  - Scores de risque basés sur l'historique

- **Gestion du Déséquilibre**
  - Technique SMOTE pour le rééchantillonnage
  - Validation stratifiée pour préserver les proportions
  - Métriques adaptées aux classes déséquilibrées

- **Performance et Production**
  - Pipeline de données automatisé
  - API REST pour les prédictions en temps réel
  - Sauvegarde des transformations pour la production
  - Monitoring des performances

## 🚀 Installation

1. Clonez le repository :
```bash
git clone https://github.com/codeGeekPro/fraud-detection-ml.git
cd fraud-detection-ml
```

2. Créez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Préparez l'environnement :
```bash
# Créez les dossiers nécessaires
mkdir -p data/raw data/processed models/trained logs

# Téléchargez le dataset (si vous avez Kaggle CLI configuré)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/
unzip data/raw/creditcardfraud.zip -d data/raw/
```

5. Configurez le projet :
```bash
# Vérifiez et ajustez la configuration dans config/config.yaml
# Les paramètres par défaut sont optimisés pour les performances

## 💻 Structure du Projet

```
fraud-detection-ml/
├── notebooks/                    # Notebooks Jupyter pour l'analyse
│   ├── 01_eda_fraud_analysis    # Analyse exploratoire des données
│   ├── 02_feature_engineering   # Développement des features
│   ├── 03_model_training       # Expérimentation des modèles
│   └── 04_model_evaluation     # Évaluation approfondie
├── src/                        # Code source principal
│   ├── data/                  # Gestion et preprocessing des données
│   │   ├── data_loader.py    # Chargement et validation des données
│   │   ├── preprocessor.py   # Pipeline de preprocessing
│   │   └── feature_engineer.py # Création de features
│   ├── models/               # Implémentation des modèles
│   │   ├── base_model.py    # Classe de base abstraite
│   │   ├── random_forest_model.py # Implémentation Random Forest
│   │   └── xgboost_model.py # Implémentation XGBoost
│   ├── utils/               # Fonctions utilitaires
│   │   ├── helpers.py      # Fonctions génériques
│   │   └── metrics.py      # Métriques d'évaluation
│   └── api/                # API REST pour les prédictions
│       ├── app.py         # Application FastAPI
│       └── prediction_service.py # Service de prédiction
├── tests/                  # Tests unitaires et d'intégration
├── config/                 # Fichiers de configuration
│   └── config.yaml        # Configuration centralisée
├── models/                 # Modèles et transformations
│   ├── trained/          # Modèles entraînés
│   └── metadata/         # Métadonnées des modèles
└── data/                  # Données du projet
    ├── raw/              # Données brutes
    ├── processed/        # Données préprocessées
    └── external/         # Données externes complémentaires
```

## 📊 Utilisation

### 1. Analyse Exploratoire
Explorez les caractéristiques des transactions et la distribution des fraudes :
```bash
jupyter notebook notebooks/01_eda_fraud_analysis.ipynb
```

### 2. Feature Engineering
Développez et testez de nouvelles features :
```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

### 3. Entraînement des Modèles
Lancez l'entraînement complet avec validation croisée :
```bash
python scripts/train_models.py
```

### 4. Évaluation des Performances
Analysez les performances détaillées des modèles :
```bash
jupyter notebook notebooks/04_model_evaluation.ipynb
```

### 5. API de Prédiction
Lancez l'API pour les prédictions en temps réel :
```bash
python src/api/app.py
```

### 6. Exemple d'Utilisation de l'API
```python
import requests

# Préparez les données de la transaction
transaction = {
    "amount": 123.45,
    "time": 43200,  # 12:00 (midi)
    "v1": -1.3598071336738172,
    "v2": -0.0727811733098497,
    "v3": 2.536346738618042,
    # ... autres features V1-V28
}

# Envoyez la requête
response = requests.post(
    "http://localhost:8000/predict",
    json={"transaction": transaction}
)

# Analysez la réponse
result = response.json()
print(f"Probabilité de fraude : {result['fraud_probability']:.2%}")
print(f"Prédiction : {'Frauduleuse' if result['is_fraud'] else 'Légitime'}")
print(f"Temps de réponse : {result['response_time_ms']:.2f}ms")
```

## 📈 Performance

Les modèles ont été évalués en utilisant la validation croisée à 5 plis :

### Random Forest (Meilleur Modèle)
- Precision-Recall AUC : 0.9999992 ± 0.0000011
- Excellente gestion des classes déséquilibrées
- Robuste aux outliers

### XGBoost
- Precision-Recall AUC : 0.9999869 ± 0.0000147
- Performance légèrement inférieure au Random Forest
- Temps d'entraînement plus court

## 🤝 Contribution

Les contributions sont les bienvenues ! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives.

## � Technologies Utilisées

- **Python 3.8+** - Langage de programmation principal
- **scikit-learn** - Algorithmes de ML et preprocessing
- **XGBoost** - Modèle de boosting
- **pandas** - Manipulation de données
- **numpy** - Calculs numériques
- **FastAPI** - API REST
- **pytest** - Tests unitaires
- **black** - Formatage de code
- **mypy** - Vérification des types
- **logging** - Journalisation structurée
- **YAML** - Configuration externalisée

## �📝 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 📧 Contact

Code Geek Pro - [GitHub](https://github.com/codeGeekPro)

Lien du projet : [https://github.com/codeGeekPro/fraud-detection-ml](https://github.com/codeGeekPro/fraud-detection-ml)
