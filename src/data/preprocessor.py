"""Preprocessor module for fraud detection."""
from typing import Optional, Dict, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

class Preprocessor(BaseEstimator, TransformerMixin):
    """Classe de prétraitement des données pour la détection de fraudes."""
    
    def __init__(self):
        """Initialisation du preprocessor."""
        pass  # À implémenter
