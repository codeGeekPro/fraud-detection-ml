"""Base model interface for fraud detection."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseModel(ABC):
    """Classe de base abstraite pour les modèles de détection de fraudes."""

    @abstractmethod
    def fit(self, X, y):
        """Entraîne le modèle."""
        pass

    @abstractmethod
    def predict(self, X):
        """Fait des prédictions."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Retourne les probabilités de prédiction."""
        pass
