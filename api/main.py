"""
main.py
-------
API de serving do modelo de previsão de doença cardíaca.

Para rodar localmente:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Documentação automática disponível em:
    http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .schemas import PatientInput, PredictionOutput
from .predictor import HeartDiseasePredictor


MODEL_PATH  = os.getenv("MODEL_PATH",  "models/mlp_heart.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

predictor: HeartDiseasePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carrega os artefatos uma vez no startup.
    Usar lifespan em vez de @app.on_event é a forma recomendada no FastAPI moderno.
    """
    global predictor
    predictor = HeartDiseasePredictor(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH
    )
    print(f"Modelo carregado: {MODEL_PATH}")
    yield
    # cleanup se necessário (fechar conexões, liberar GPU, etc.)


app = FastAPI(
    title="Heart Disease Predictor",
    description="Previsão de doença cardíaca a partir de variáveis clínicas.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    """Endpoint de health check — usado por load balancers e orquestradores."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    """
    Recebe os dados clínicos de um paciente e retorna a probabilidade
    de doença cardíaca e o diagnóstico binário.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    try:
        result = predictor.predict(patient.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)
