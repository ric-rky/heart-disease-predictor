"""
preprocessing.py
----------------
Funções de pré-processamento para o dataset Heart Disease UCI.
Tudo aqui é stateless em relação ao notebook: recebe DataFrames, devolve DataFrames
ou objetos de transformação já ajustados.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Colunas do dataset UCI Heart Disease (Cleveland)
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Features numéricas contínuas que vão receber StandardScaler
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Features categóricas (one-hot encoding)
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]

# Features binárias (mantidas como estão)
BINARY_FEATURES = ["sex", "fbs", "exang"]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o CSV do UCI, adiciona nomes de colunas e remove linhas com '?'.
    O dataset original usa '?' para missing values em 'ca' e 'thal'.
    """
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES, na_values="?")
    n_antes = len(df)
    df = df.dropna().reset_index(drop=True)
    n_depois = len(df)
    print(f"Linhas carregadas: {n_antes} | Após remoção de NaN: {n_depois}")
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    O target original vai de 0 a 4.
    Binarizamos: 0 = sem doença, 1 = com doença (qualquer grau > 0).
    """
    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)
    print(f"Distribuição do target:\n{df['target'].value_counts(normalize=True).round(3)}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica one-hot encoding nas features categóricas, mantendo binárias e contínuas.
    Devolve um DataFrame novo sem a coluna target.
    """
    df = df.copy()

    # One-hot nas categóricas
    df_ohe = pd.get_dummies(
        df[CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        prefix=CATEGORICAL_FEATURES,
        drop_first=False,       # manter todas as categorias facilita interpretação
        dtype=float
    )

    df_final = pd.concat([
        df[CONTINUOUS_FEATURES].astype(float),
        df[BINARY_FEATURES].astype(float),
        df_ohe
    ], axis=1)

    return df_final


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    """
    Divide em treino/validação/teste e aplica StandardScaler apenas nas features
    contínuas, ajustado exclusivamente no treino (sem data leakage).

    Retorna:
        X_train, X_val, X_test (np.ndarray)
        y_train, y_val, y_test (np.ndarray)
        scaler (StandardScaler ajustado)
        feature_names (list)
    """
    # Primeiro split: separa teste
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Segundo split: separa validação do treino
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_trainval
    )

    feature_names = list(X.columns)
    continuous_idx = [feature_names.index(c) for c in CONTINUOUS_FEATURES]

    # Scaler ajustado só no treino
    scaler = StandardScaler()
    X_train_arr = X_train.values.copy()
    X_val_arr   = X_val.values.copy()
    X_test_arr  = X_test.values.copy()

    X_train_arr[:, continuous_idx] = scaler.fit_transform(X_train_arr[:, continuous_idx])
    X_val_arr[:, continuous_idx]   = scaler.transform(X_val_arr[:, continuous_idx])
    X_test_arr[:, continuous_idx]  = scaler.transform(X_test_arr[:, continuous_idx])

    print(f"Treino:    {X_train_arr.shape[0]} amostras")
    print(f"Validação: {X_val_arr.shape[0]} amostras")
    print(f"Teste:     {X_test_arr.shape[0]} amostras")

    return (
        X_train_arr, X_val_arr, X_test_arr,
        y_train.values, y_val.values, y_test.values,
        scaler, feature_names
    )
