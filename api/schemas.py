"""
schemas.py
----------
Modelos Pydantic para validação de entrada e saída da API.
Separar schemas de lógica de negócio é boa prática: permite versionar
a API sem mexer no predictor, e facilita testes unitários dos endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class PatientInput(BaseModel):
    age:      float = Field(..., ge=1,  le=120, description="Idade em anos")
    sex:      int   = Field(..., ge=0,  le=1,   description="Sexo (1=masculino, 0=feminino)")
    cp:       int   = Field(..., ge=0,  le=3,   description="Tipo de dor no peito (0–3)")
    trestbps: float = Field(..., ge=80, le=250, description="Pressão arterial em repouso (mmHg)")
    chol:     float = Field(..., ge=100,le=600, description="Colesterol sérico (mg/dl)")
    fbs:      int   = Field(..., ge=0,  le=1,   description="Glicemia em jejum > 120 mg/dl")
    restecg:  int   = Field(..., ge=0,  le=2,   description="Resultado do ECG em repouso (0–2)")
    thalach:  float = Field(..., ge=60, le=250, description="Frequência cardíaca máxima")
    exang:    int   = Field(..., ge=0,  le=1,   description="Angina induzida por exercício")
    oldpeak:  float = Field(..., ge=0,  le=10,  description="Depressão do segmento ST")
    slope:    int   = Field(..., ge=0,  le=2,   description="Inclinação do segmento ST (0–2)")
    ca:       int   = Field(..., ge=0,  le=3,   description="Número de vasos coloridos (0–3)")
    thal:     int   = Field(..., description="Talassemia (3=normal, 6=defeito fixo, 7=defeito reversível)")

    @field_validator("thal")
    @classmethod
    def thal_valido(cls, v):
        if v not in (3, 6, 7):
            raise ValueError("thal deve ser 3, 6 ou 7")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
                "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
                "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 7
            }
        }
    }


class PredictionOutput(BaseModel):
    probabilidade_doenca: float
    diagnostico:          Literal["Com doença", "Sem doença"]
    threshold_usado:      float
