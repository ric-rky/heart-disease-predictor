"""
persistence.py
--------------
Funções para salvar e carregar modelo treinado e scaler.
Em produção, esses artefatos versionados ficam num bucket (S3, GCS)
ou num model registry (MLflow, W&B). Aqui usamos disco local.
"""

import os
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def save_model(model: nn.Module, filepath: str):
    """
    Salva apenas os pesos (state_dict), não a arquitetura.
    Para carregar, a arquitetura precisa ser instanciada primeiro.
    Isso é preferível a torch.save(model) que pickla a classe inteira
    e quebra quando o código de definição muda.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Pesos salvos em: {filepath}")


def load_model(model: nn.Module, filepath: str, device: str = "cpu") -> nn.Module:
    """
    Carrega pesos num modelo já instanciado.
    """
    state_dict = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Pesos carregados de: {filepath}")
    return model


def save_scaler(scaler: StandardScaler, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler salvo em: {filepath}")


def load_scaler(filepath: str) -> StandardScaler:
    with open(filepath, "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler carregado de: {filepath}")
    return scaler
