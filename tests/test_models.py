"""Tests for model modules."""

import pytest
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.ensemble_model import EnsembleModel

import numpy as np
import pandas as pd
import tempfile


def get_sample_data():
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    y = np.array([0, 1, 0, 1, 0])
    return X, y


def test_random_forest_model():
    X, y = get_sample_data()
    model = RandomForestModel({"n_estimators": 10, "random_state": 42})
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    probas = model.predict_proba(X)
    assert probas.shape[0] == len(y)
    # Test save/load
    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        model.save(tmp.name)
        model2 = RandomForestModel()
        model2.load(tmp.name)
        preds2 = model2.predict(X)
        assert np.array_equal(preds, preds2)


def test_xgboost_model():
    X, y = get_sample_data()
    model = XGBoostModel(
        {
            "n_estimators": 10,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    probas = model.predict_proba(X)
    assert probas.shape[0] == len(y)
    # Test save/load
    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        model.save(tmp.name)
        model2 = XGBoostModel()
        model2.load(tmp.name)
        preds2 = model2.predict(X)
        assert np.array_equal(preds, preds2)
