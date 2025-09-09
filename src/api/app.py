"""
API FastAPI pour la détection de fraude.
Endpoints : /predict, /predict_proba, /model_info, /health
Validation des inputs avec Pydantic, logging complet.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import time
from pathlib import Path
import sys

# Ajout du chemin racine au sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.api.prediction_service import PredictionService

app = FastAPI(
    title="Fraud Detection API",
    description="API pour la détection de transactions frauduleuses en temps réel.",
    version="1.0.0"
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Input schema pour les transactions
class TransactionInput(BaseModel):
    """
    Schéma d'entrée pour une transaction bancaire.

    Attributes:
        Time (float): Temps écoulé depuis la première transaction
        V1 to V28 (float): Features anonymisées
        Amount (float): Montant de la transaction
    """
    Time: float = Field(..., description="Temps écoulé depuis la première transaction")
    V1: float = Field(..., description="Feature anonymisée V1")
    V2: float = Field(..., description="Feature anonymisée V2")
    V3: float = Field(..., description="Feature anonymisée V3")
    V4: float = Field(..., description="Feature anonymisée V4")
    V5: float = Field(..., description="Feature anonymisée V5")
    V6: float = Field(..., description="Feature anonymisée V6")
    V7: float = Field(..., description="Feature anonymisée V7")
    V8: float = Field(..., description="Feature anonymisée V8")
    V9: float = Field(..., description="Feature anonymisée V9")
    V10: float = Field(..., description="Feature anonymisée V10")
    V11: float = Field(..., description="Feature anonymisée V11")
    V12: float = Field(..., description="Feature anonymisée V12")
    V13: float = Field(..., description="Feature anonymisée V13")
    V14: float = Field(..., description="Feature anonymisée V14")
    V15: float = Field(..., description="Feature anonymisée V15")
    V16: float = Field(..., description="Feature anonymisée V16")
    V17: float = Field(..., description="Feature anonymisée V17")
    V18: float = Field(..., description="Feature anonymisée V18")
    V19: float = Field(..., description="Feature anonymisée V19")
    V20: float = Field(..., description="Feature anonymisée V20")
    V21: float = Field(..., description="Feature anonymisée V21")
    V22: float = Field(..., description="Feature anonymisée V22")
    V23: float = Field(..., description="Feature anonymisée V23")
    V24: float = Field(..., description="Feature anonymisée V24")
    V25: float = Field(..., description="Feature anonymisée V25")
    V26: float = Field(..., description="Feature anonymisée V26")
    V27: float = Field(..., description="Feature anonymisée V27")
    V28: float = Field(..., description="Feature anonymisée V28")
    Amount: float = Field(..., description="Montant de la transaction")


# Output schema
class PredictionOutput(BaseModel):
    """
    Schéma de sortie pour la prédiction.

    Attributes:
        prediction (int): Classe prédite (0: légitime, 1: frauduleuse)
        fraud_probability (float): Probabilité de fraude
        is_fraud (bool): Indique si la transaction est frauduleuse
        confidence (float): Niveau de confiance de la prédiction
        risk_level (str): Niveau de risque (TRÈS FAIBLE, FAIBLE, MOYEN, ÉLEVÉ, TRÈS ÉLEVÉ)
        response_time_ms (float): Temps de réponse en millisecondes
        model_version (str): Version du modèle utilisé
        timestamp (str): Timestamp de la prédiction
    """
    prediction: int
    fraud_probability: float
    is_fraud: bool
    confidence: float
    risk_level: str
    response_time_ms: float
    model_version: str
    timestamp: str


# Batch input schema
class BatchTransactionInput(BaseModel):
    """
    Schéma d'entrée pour un lot de transactions.
    """
    transactions: List[TransactionInput] = Field(..., description="Liste des transactions à analyser")


# Batch output schema
class BatchPredictionOutput(BaseModel):
    """
    Schéma de sortie pour les prédictions par lot.
    """
    predictions: List[PredictionOutput]
    total_transactions: int
    fraud_detected: int
    processing_time_ms: float


# Initialisation du service
service = PredictionService()


@app.get("/health")
def health() -> dict:
    """
    Vérifie l'état de l'API.

    Returns:
        dict: Statut de l'API et informations système.
    """
    logger.info("Health check requested.")
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "fraud-detection-api",
        "version": "1.0.0"
    }


@app.get("/model_info")
def model_info() -> dict:
    """
    Retourne les informations détaillées du modèle.

    Returns:
        dict: Informations complètes sur le modèle.
    """
    info = service.get_model_info()
    logger.info("Model info requested.")
    return info


@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput) -> PredictionOutput:
    """
    Prédit si une transaction est frauduleuse.

    Args:
        transaction (TransactionInput): Données de la transaction.

    Returns:
        PredictionOutput: Résultat détaillé de la prédiction.
    """
    start_time = time.time()

    try:
        result = service.predict_single(transaction.dict())
        result["response_time_ms"] = (time.time() - start_time) * 1000
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(".2f")

        return PredictionOutput(**result)

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchTransactionInput) -> BatchPredictionOutput:
    """
    Prédit pour un lot de transactions.

    Args:
        batch (BatchTransactionInput): Lot de transactions.

    Returns:
        BatchPredictionOutput: Résultats pour toutes les transactions.
    """
    start_time = time.time()

    try:
        results = service.predict_batch([t.dict() for t in batch.transactions])
        processing_time = (time.time() - start_time) * 1000

        fraud_count = sum(1 for r in results if r.get("is_fraud", False))

        logger.info(f"Batch prediction: {len(results)} transactions, {fraud_count} frauds detected")

        return BatchPredictionOutput(
            predictions=[PredictionOutput(**r) for r in results],
            total_transactions=len(results),
            fraud_detected=fraud_count,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Erreur lors du traitement par lot : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement par lot : {str(e)}")


@app.get("/stats")
def get_stats() -> dict:
    """
    Retourne les statistiques d'utilisation de l'API.

    Returns:
        dict: Statistiques d'utilisation.
    """
    # Dans une vraie implémentation, ces stats seraient stockées en base
    return {
        "total_predictions": 0,
        "fraud_predictions": 0,
        "average_response_time_ms": 0.0,
        "uptime_seconds": 0
    }
