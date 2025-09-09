"""FastAPI application for fraud detection."""
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
    """Modèle Pydantic pour une transaction."""
    pass  # À implémenter

@app.post("/predict")
async def predict(transaction: Transaction) -> Dict[str, Any]:
    """Endpoint pour la prédiction de fraude."""
    pass  # À implémenter
