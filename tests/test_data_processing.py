"""Tests for data processing modules."""

import pytest
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.feature_engineer import FeatureEngineer


def test_data_loader():
    """Test du module de chargement des données."""
    pass  # À implémenter


import pandas as pd
import numpy as np


def get_sample_df():
    return pd.DataFrame(
        {
            "Time": [1000, 2000, 3000, 4000, 5000],
            "Amount": [10, 20, 30, 40, 50],
            "UserID": ["u1", "u2", "u1", "u2", "u1"],
            "MerchantID": ["m1", "m2", "m1", "m2", "m1"],
            "Class": [0, 1, 0, 1, 0],
            "Category": ["A", "B", "A", "B", "C"],
        }
    )


def test_feature_engineer_temporal():
    df = get_sample_df()
    fe = FeatureEngineer()
    out = fe.add_temporal_features(df.copy())
    assert "transaction_hour" in out.columns
    assert "transaction_day" in out.columns
    assert "transaction_month" in out.columns


def test_feature_engineer_zscore():
    df = get_sample_df()
    fe = FeatureEngineer()
    out = fe.add_amount_zscore_by_user(df.copy())
    assert "amount_zscore_by_user" in out.columns


def test_feature_engineer_frequency():
    df = get_sample_df()
    fe = FeatureEngineer()
    out = fe.add_frequency_features(df.copy())
    assert "frequency_last_1h" in out.columns
    assert "frequency_last_24h" in out.columns


def test_feature_engineer_deviation():
    df = get_sample_df()
    fe = FeatureEngineer()
    out = fe.add_amount_deviation_from_avg(df.copy())
    assert "amount_deviation_from_avg" in out.columns


def test_feature_engineer_risk_score():
    df = get_sample_df()
    fe = FeatureEngineer()
    out = fe.add_merchant_risk_score(df.copy())
    assert "merchant_risk_score" in out.columns


def test_feature_engineer_time_since_last():
    df = get_sample_df()
    fe = FeatureEngineer()
    out = fe.add_time_since_last_transaction(df.copy())
    assert "time_since_last_transaction" in out.columns


def test_feature_engineer_velocity():
    df = get_sample_df()
    fe = FeatureEngineer()
    out = fe.add_transaction_velocity(df.copy())
    assert "transaction_velocity" in out.columns


def test_feature_engineer_encoding():
    df = get_sample_df()
    fe = FeatureEngineer()
    out = fe.encode_categorical(df.copy(), "Category", encoding="onehot")
    assert any(["Category_" in c for c in out.columns])
    out2 = fe.encode_categorical(df.copy(), "Category", encoding="target")
    assert "Category_target_enc" in out2.columns
    out3 = fe.encode_categorical(df.copy(), "Category", encoding="ordinal")
    assert "Category_ordinal_enc" in out3.columns


def test_feature_engineer_fit_transform():
    df = get_sample_df()
    fe = FeatureEngineer()
    out = fe.fit_transform(df.copy())
    # Vérifie que toutes les features principales sont présentes
    for col in [
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "amount_zscore_by_user",
        "frequency_last_1h",
        "frequency_last_24h",
        "amount_deviation_from_avg",
        "merchant_risk_score",
        "time_since_last_transaction",
        "transaction_velocity",
    ]:
        assert col in out.columns


def test_preprocessor_mean():
    df = get_sample_df()
    pre = Preprocessor(missing_strategy="mean", scaling_strategy="standard")
    arr = pre.fit_transform(df)
    assert arr.shape == df.shape
    num_cols = arr.select_dtypes(include=[np.number]).columns
    assert not arr[num_cols].isnull().any().any()


def test_preprocessor_median():
    df = get_sample_df()
    pre = Preprocessor(missing_strategy="median", scaling_strategy="minmax")
    arr = pre.fit_transform(df)
    assert arr.shape == df.shape
    num_cols = arr.select_dtypes(include=[np.number]).columns
    assert not arr[num_cols].isnull().any().any()


def test_preprocessor_mode():
    df = get_sample_df()
    pre = Preprocessor(missing_strategy="mode", scaling_strategy="robust")
    arr = pre.fit_transform(df)
    assert arr.shape == df.shape
    num_cols = arr.select_dtypes(include=[np.number]).columns
    assert not arr[num_cols].isnull().any().any()


def test_preprocessor_knn():
    df = get_sample_df()
    pre = Preprocessor(missing_strategy="knn", scaling_strategy=None, knn_neighbors=2)
    arr = pre.fit_transform(df)
    assert arr.shape == df.shape
    num_cols = arr.select_dtypes(include=[np.number]).columns
    assert not arr[num_cols].isnull().any().any()


def test_preprocessor_pipeline():
    df = get_sample_df()
    pre = Preprocessor(missing_strategy="median", scaling_strategy="standard")
    pre.fit(df)
    arr = pre.transform(df)
    assert arr.shape == df.shape


def test_preprocessor_feature_names():
    df = get_sample_df()
    pre = Preprocessor()
    names = pre.get_feature_names(df)
    assert names == list(df.columns)
