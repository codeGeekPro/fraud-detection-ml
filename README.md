# 🔍 Détection de Fraudes Bancaires avec Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen.svg)](https://github.com/username/fraud-detection-ml)

## 📋 Description

Système de détection de fraudes bancaires basé sur le Machine Learning, utilisant des algorithmes avancés (Random Forest et XGBoost) pour identifier automatiquement les transactions frauduleuses. Le projet met l'accent sur la gestion efficace des données déséquilibrées et l'interprétabilité des résultats.

### 🎯 Caractéristiques Principales

- Détection temps réel des transactions frauduleuses
- Gestion avancée des données déséquilibrées
- Feature engineering temporel sophistiqué
- API REST pour les prédictions en production
- Visualisations interactives avec Plotly
- Tests unitaires complets
- Documentation détaillée

## 🚀 Installation

1. Clonez le repository :
```bash
git clone https://github.com/username/fraud-detection-ml.git
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

4. Téléchargez le dataset :
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/
```

## 💻 Structure du Projet

```
fraud-detection-ml/
├── notebooks/          # Notebooks Jupyter pour l'analyse
├── src/               # Code source principal
│   ├── data/         # Gestion et preprocessing des données
│   ├── models/       # Implémentation des modèles
│   ├── utils/        # Fonctions utilitaires
│   └── api/          # API REST pour les prédictions
├── tests/            # Tests unitaires et d'intégration
├── config/           # Fichiers de configuration
└── models/           # Modèles entraînés et métadonnées
```

## 📊 Utilisation

1. Exploration des données :
```bash
jupyter notebook notebooks/01_eda_fraud_analysis.ipynb
```

2. Entraînement des modèles :
```bash
python scripts/train_models.py
```

3. Lancement de l'API :
```bash
python src/api/app.py
```

4. Exemple de prédiction via l'API :
```python
import requests

data = {
    "amount": 123.45,
    "time": 0,
    "v1": 0.123,
    # ... autres features
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
```

## 📈 Performance

- Precision-Recall AUC : 0.85
- F1-Score : 0.82
- Matthews Correlation Coefficient : 0.79

## 🤝 Contribution

Les contributions sont les bienvenues ! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives.

## 📝 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 📧 Contact

Votre Nom - email@example.com

Lien du projet : [https://github.com/username/fraud-detection-ml](https://github.com/username/fraud-detection-ml)
