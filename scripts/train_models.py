"""Script d'entraînement des modèles."""

import sys
from pathlib import Path

# Ajout du chemin racine au sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import argparse
import logging
from src.utils.helpers import setup_logging, load_config
from src.data.data_loader import DataLoader
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel


def main() -> None:
    """
    Point d'entrée principal pour l'entraînement des modèles.

    - Charge la config et le dataset
    - Applique le préprocessing et le feature engineering
    - Gère le déséquilibre
    - Sépare les données
    - Entraîne Random Forest et XGBoost avec validation croisée
    - Sauvegarde le meilleur modèle
    """
    # Chargement de la config
    config = load_config("config/config.yaml")

    # Setup logging à partir du dict
    log_cfg = config["logging"]
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
        filename=log_cfg.get("file_path"),
    )
    logger = logging.getLogger("train_models")

    # Chargement des données
    data_loader = DataLoader()
    df = data_loader.load_data(file_path="data/raw/creditcard.csv")
    logger.info(f"Données chargées : {df.shape}")

    # Préprocessing
    from src.data.preprocessor import Preprocessor

    # Extraire uniquement les valeurs nécessaires pour le préprocesseur
    preprocessing_config = config["preprocessing"]
    missing_strategy = preprocessing_config["missing_values_strategy"]
    scaling_strategy = preprocessing_config["scaling_method"]

    preprocessor = Preprocessor(
        missing_strategy=missing_strategy,
        scaling_strategy=scaling_strategy
    )

    df_prep = preprocessor.fit_transform(df)
    logger.info(f"Données prétraitées : {df_prep.shape}")

    # Feature engineering
    from src.data.feature_engineer import FeatureEngineer
    feature_engineer = FeatureEngineer(config["feature_engineering"])
    df_feat = feature_engineer.fit_transform(df_prep)
    logger.info(f"Features créées : {df_feat.shape}")

    # Séparation train/val/test
    from sklearn.model_selection import train_test_split, StratifiedKFold
    X = df_feat.drop("Class", axis=1)
    y = df_feat["Class"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config["preprocessing"]["test_size"] + config["preprocessing"]["val_size"],
        stratify=y, random_state=config["preprocessing"]["random_state"]
    )
    val_ratio = config["preprocessing"]["val_size"] / (config["preprocessing"]["test_size"] + config["preprocessing"]["val_size"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_ratio,
        stratify=y_temp, random_state=config["preprocessing"]["random_state"]
    )
    logger.info(f"Split : train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Gestion du déséquilibre (SMOTE)
    if config["training"]["resampling_strategy"] == "smote":
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=config["preprocessing"]["random_state"])
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"SMOTE appliqué : train={X_train.shape}")

    # Entraînement Random Forest
    rf = RandomForestModel(config["model"]["random_forest"])
    rf.fit(X_train, y_train)
    logger.info("Random Forest entraîné.")

    # Entraînement XGBoost
    xgb = XGBoostModel(config["model"]["xgboost"])
    xgb.fit(X_train, y_train)
    logger.info("XGBoost entraîné.")

    # Validation croisée stratifiée
    skf = StratifiedKFold(n_splits=config["training"]["cv_folds"], shuffle=True, random_state=config["preprocessing"]["random_state"])
    from src.utils.metrics import pr_auc_score
    rf_scores = []
    xgb_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        rf.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)
        rf_pred = rf.predict_proba(X_va)[:, 1]
        xgb_pred = xgb.predict_proba(X_va)[:, 1]
        rf_scores.append(pr_auc_score(y_va, rf_pred))
        xgb_scores.append(pr_auc_score(y_va, xgb_pred))
    logger.info(f"RF PR-AUC CV : {rf_scores}")
    logger.info(f"XGB PR-AUC CV : {xgb_scores}")

    # Sélection du meilleur modèle
    if sum(rf_scores) / len(rf_scores) > sum(xgb_scores) / len(xgb_scores):
        best_model = rf
        logger.info("Random Forest sélectionné.")
    else:
        best_model = xgb
        logger.info("XGBoost sélectionné.")

    # Sauvegarde du modèle
    try:
        import joblib
        joblib.dump(best_model, config["api"]["model_path"])
        logger.info(f"Modèle sauvegardé dans {config['api']['model_path']}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modèle : {e}")
        raise


if __name__ == "__main__":
    main()
