"""
app.py
------
Interface Streamlit para demonstração do modelo de previsão de doença cardíaca.

Para rodar:
    streamlit run app/app.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.predictor import HeartDiseasePredictor

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

MODEL_PATH  = os.getenv("MODEL_PATH",  "models/mlp_heart.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")


# ---------------------------------------------------------------------------
# Carrega o modelo uma vez por sessão
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    return HeartDiseasePredictor(MODEL_PATH, SCALER_PATH)


try:
    predictor = load_predictor()
    modelo_carregado = True
except Exception as e:
    modelo_carregado = False
    erro_carregamento = str(e)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Previsão de Doença Cardíaca")
st.markdown(
    "Preencha os dados clínicos do paciente e clique em **Analisar** "
    "para obter a probabilidade de doença cardíaca."
)

if not modelo_carregado:
    st.error(
        f"Modelo não encontrado. Execute o notebook para gerar os artefatos "
        f"em `models/`. Detalhe: {erro_carregamento}"
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Formulário de entrada
# ---------------------------------------------------------------------------
col_esq, col_dir = st.columns(2)

with col_esq:
    st.subheader("Dados demográficos e vitais")

    age      = st.slider("Idade (anos)",            min_value=20,  max_value=100, value=52)
    sex      = st.radio("Sexo", options=[1, 0],     format_func=lambda x: "Masculino" if x == 1 else "Feminino", horizontal=True)
    trestbps = st.slider("Pressão arterial em repouso (mmHg)", min_value=80, max_value=220, value=130)
    chol     = st.slider("Colesterol sérico (mg/dl)",          min_value=100, max_value=570, value=240)
    thalach  = st.slider("Frequência cardíaca máxima",         min_value=70, max_value=210, value=150)
    fbs      = st.radio(
        "Glicemia em jejum > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "Não" if x == 0 else "Sim",
        horizontal=True
    )

with col_dir:
    st.subheader("Dados do exame")

    cp = st.selectbox(
        "Tipo de dor no peito",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0: Angina típica",
            1: "1: Angina atípica",
            2: "2: Dor não-anginosa",
            3: "3: Assintomático"
        }[x]
    )
    restecg = st.selectbox(
        "Resultado do ECG em repouso",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0: Normal",
            1: "1: Anormalidade ST-T",
            2: "2: Hipertrofia ventricular esquerda"
        }[x]
    )
    exang = st.radio(
        "Angina induzida por exercício",
        options=[0, 1],
        format_func=lambda x: "Não" if x == 0 else "Sim",
        horizontal=True
    )
    oldpeak = st.slider("Depressão do segmento ST",    min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope   = st.selectbox(
        "Inclinação do segmento ST no pico",
        options=[0, 1, 2],
        format_func=lambda x: {0: "0: Ascendente", 1: "1: Plano", 2: "2: Descendente"}[x]
    )
    ca   = st.selectbox("Número de vasos principais (fluoroscopia)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Talassemia",
        options=[3, 6, 7],
        format_func=lambda x: {3: "3: Normal", 6: "6: Defeito fixo", 7: "7: Defeito reversível"}[x]
    )

st.divider()

# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------
if st.button("Analisar", type="primary", use_container_width=True):
    dados = dict(
        age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
        fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
        oldpeak=oldpeak, slope=slope, ca=ca, thal=thal
    )

    with st.spinner("Calculando..."):
        resultado = predictor.predict(dados)

    prob       = resultado["probabilidade_doenca"]
    diagnostico = resultado["diagnostico"]

    st.divider()
    st.subheader("Resultado")

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.metric("Probabilidade de doença", f"{prob:.1%}")

    with col_res2:
        if diagnostico == "Com doença":
            st.error(f"**{diagnostico}**")
        else:
            st.success(f"**{diagnostico}**")

    with col_res3:
        st.metric("Threshold de decisão", f"{resultado['threshold_usado']:.2f}")

    # Gauge simples via matplotlib
    fig, ax = plt.subplots(figsize=(4, 0.5))
    cor = "#DC2626" if prob >= 0.5 else "#16A34A"
    ax.barh([""], [prob],       color=cor,     height=0.5, alpha=0.85)
    ax.barh([""], [1 - prob],   left=[prob],   color="#E5E7EB", height=0.5)
    ax.axvline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption(
        "Este modelo é uma demonstração. "
        "Não substitui avaliação médica profissional."
    )
