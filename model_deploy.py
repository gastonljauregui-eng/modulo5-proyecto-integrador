import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any


app = FastAPI(
    title="API de Predicción - Proyecto Integrador",
    description="API para disponibilizar el modelo entrenado mediante un endpoint de predicción.",
    version="1.0.0"
)


model = joblib.load("models/best_model.pkl")


class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]


@app.get("/")
def home():
    return {
        "message": "API funcionando correctamente",
        "endpoint": "/predict"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    df = pd.DataFrame(request.data)

    predictions = model.predict(df)

    return {
        "predictions": predictions.tolist()
    }