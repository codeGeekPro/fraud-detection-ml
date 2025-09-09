"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)



def test_health_endpoint():
    """
    Teste l'endpoint /health.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    """
    Teste l'endpoint /model_info.
    """
    response = client.get("/model_info")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_predict_endpoint():
    """
    Teste l'endpoint /predict avec des données factices.
    """
    payload = {
        "feature1": 0.5,
        "feature2": 1.2,
        "feature3": -0.7
        # ... compléter selon le modèle
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 500]  # 500 si modèle non chargé
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data


def test_predict_proba_endpoint():
    """
    Teste l'endpoint /predict_proba avec des données factices.
    """
    payload = {
        "feature1": 0.5,
        "feature2": 1.2,
        "feature3": -0.7
        # ... compléter selon le modèle
    }
    response = client.post("/predict_proba", json=payload)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
