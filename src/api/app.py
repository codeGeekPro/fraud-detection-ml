"""
API FastAPI pour la détection de fraude.
Endpoints : /predict, /predict_proba, /model_info, /health
Validation des inputs avec Pydantic, logging complet.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from src.api.prediction_service import PredictionService

app = FastAPI(
    title="Fraud Detection API",
    description="API pour la détection de transactions frauduleuses.",
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Input schema (à adapter selon vos features réelles)
class TransactionInput(BaseModel):
    """
    Schéma d'entrée pour une transaction.

    Attributes:
        feature1 (float): Première feature.
        feature2 (float): Deuxième feature.
        feature3 (float): Troisième feature.
        # ... ajouter les autres features nécessaires
    """
    feature1: float
    feature2: float
    feature3: float
    # ... ajouter les autres features nécessaires


# Output schema
class PredictionOutput(BaseModel):
    """
    Schéma de sortie pour la prédiction.

    Attributes:
        prediction (int): Classe prédite (0 ou 1).
        probability (float): Probabilité de fraude.
    """
    prediction: int
    probability: float


service = PredictionService()


@app.get("/health")
def health() -> dict:
    """
    Vérifie l'état de l'API.

    Returns:
        dict: Statut de l'API.
    """
    logger.info("Health check requested.")
    return {"status": "ok"}


@app.get("/model_info")
def model_info() -> dict:
    """
    Retourne les infos du modèle.

    Returns:
        dict: Informations sur le modèle.
    """
    info = service.get_model_info()
    logger.info("Model info requested.")
    return info


@app.post("/predict", response_model=PredictionOutput)
def predict(input: TransactionInput) -> PredictionOutput:
    """
    Prédit la classe (fraude ou non) pour une transaction.

    Args:
        input (TransactionInput): Données de la transaction.

    Returns:
        PredictionOutput: Résultat de la prédiction.
    """
    try:
        pred, proba = service.predict(input.dict())
        logger.info(f"Prediction requested: {input.dict()} => {pred}, {proba}")
        return PredictionOutput(prediction=pred, probability=proba)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_proba", response_model=PredictionOutput)
def predict_proba(input: TransactionInput) -> PredictionOutput:
    """
    Retourne la probabilité de fraude pour une transaction.

    Args:
        input (TransactionInput): Données de la transaction.

    Returns:
        PredictionOutput: Résultat de la prédiction (probabilité).
    """
    try:
        pred, proba = service.predict_proba(input.dict())
        logger.info(f"Probability requested: {input.dict()} => {pred}, {proba}")
        return PredictionOutput(prediction=pred, probability=proba)
    except Exception as e:
        logger.error(f"Probability error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
