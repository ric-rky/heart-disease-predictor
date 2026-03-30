"""
predictor.py
------------
Carrega os artefatos uma única vez na inicialização da aplicação e expõe
um método de inferência. Manter isso separado do main.py facilita testar
a lógica de predição sem subir o servidor HTTP.
"""

import sys
import os
import pickle
import numpy as np
import torch

# Adiciona src ao path para reusar os módulos do projeto
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model import MLP
from preprocessing import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES


# Configuração do modelo: deve bater com o que foi usado no treino
HIDDEN_DIMS  = [64, 32]
DROPOUT_RATE = 0.3
THRESHOLD    = 0.5

# Colunas contínuas que recebem scaling (mesma ordem do preprocessing.py)
N_CONTINUOUS = len(CONTINUOUS_FEATURES)


class HeartDiseasePredictor:
    def __init__(self, model_path: str, scaler_path: str):
        self.threshold = THRESHOLD
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"

        # Carrega scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Inferimos input_dim a partir do scaler + encoding esperado
        # Continuous (5) + Binary (3) + OHE das categóricas
        # cp(4) + restecg(3) + slope(3) + ca(4) + thal(3) = 17
        input_dim = N_CONTINUOUS + len(BINARY_FEATURES) + 17

        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=HIDDEN_DIMS,
            dropout_rate=DROPOUT_RATE
        )
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, data: dict) -> np.ndarray:
        """
        Replica o encode_features do preprocessing.py para uma única amostra.
        A ordem das features precisa ser idêntica ao que o modelo viu no treino.
        """
        continuous = [
            data["age"], data["trestbps"], data["chol"],
            data["thalach"], data["oldpeak"]
        ]
        binary = [data["sex"], data["fbs"], data["exang"]]

        # One-hot manual — mesma lógica do pd.get_dummies com drop_first=False
        cp_ohe       = [int(data["cp"] == i)       for i in range(4)]
        restecg_ohe  = [int(data["restecg"] == i)  for i in range(3)]
        slope_ohe    = [int(data["slope"] == i)    for i in range(3)]
        ca_ohe       = [int(data["ca"] == i)       for i in range(4)]
        thal_ohe     = [int(data["thal"] == v)     for v in (3, 6, 7)]

        features = (
            continuous + binary +
            cp_ohe + restecg_ohe + slope_ohe + ca_ohe + thal_ohe
        )
        return np.array(features, dtype=float)

    def predict(self, data: dict) -> dict:
        features = self._encode(data)

        # Aplica scaling só nas features contínuas (primeiras N_CONTINUOUS)
        features[:N_CONTINUOUS] = self.scaler.transform(
            features[:N_CONTINUOUS].reshape(1, -1)
        ).flatten()

        with torch.no_grad():
            x      = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            prob   = torch.sigmoid(self.model(x)).item()

        return {
            "probabilidade_doenca": round(prob, 4),
            "diagnostico":          "Com doença" if prob >= self.threshold else "Sem doença",
            "threshold_usado":      self.threshold
        }
