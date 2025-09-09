"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_predict_endpoint():
    """Test de l'endpoint de prédiction."""
    pass  # À implémenter
