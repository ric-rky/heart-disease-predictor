FROM python:3.11-slim

WORKDIR /app

# Dependências primeiro para aproveitar cache de camadas
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código e artefatos
COPY src/       ./src/
COPY api/       ./api/
COPY models/    ./models/

ENV MODEL_PATH=models/mlp_heart.pt
ENV SCALER_PATH=models/scaler.pkl

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
