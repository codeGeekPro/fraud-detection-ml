"""Tests avancés pour l'évaluation (SHAP, comparaison multi-modèles)."""

import numpy as np
import pandas as pd
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.utils.evaluator import Evaluator


def get_sample_data():
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    y = np.array([0, 1, 0, 1, 0])
    return X, y


def test_compare_models():
    X, y = get_sample_data()
    rf = RandomForestModel({"n_estimators": 10, "random_state": 42})
    xgb = XGBoostModel(
        {
            "n_estimators": 10,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
    )
    rf.fit(X, y)
    xgb.fit(X, y)
    evaluator = Evaluator(y, rf.predict(X), rf.predict_proba(X))
    models = {"RandomForest": rf, "XGBoost": xgb}
    df = evaluator.compare_models(models, X, y)
    assert "f1" in df.columns
    assert "RandomForest" in df.index
    assert "XGBoost" in df.index


# Le test SHAP est illustratif (affichage graphique, non asserté)
def test_shap_summary_plot():
    X, y = get_sample_data()
    rf = RandomForestModel({"n_estimators": 10, "random_state": 42})
    rf.fit(X, y)
    evaluator = Evaluator(y, rf.predict(X), rf.predict_proba(X))
    # Ne lève pas d'exception
    evaluator.shap_summary_plot(rf.model, X)
