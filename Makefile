VENV  := .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

.DEFAULT_GOAL := help

.PHONY: help setup notebook api app docker clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make setup     Cria o ambiente virtual e instala as dependências"
	@echo "  make notebook  Abre o JupyterLab"
	@echo "  make api       Sobe a API FastAPI (porta 8000)"
	@echo "  make app       Sobe a interface Streamlit (porta 8501)"
	@echo "  make docker    Build e run da imagem Docker da API"
	@echo "  make clean     Remove o ambiente virtual"

setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements.lock
	@echo ""
	@echo "Ambiente pronto. Para ativar manualmente: source $(VENV)/bin/activate"

notebook: setup
	$(VENV)/bin/jupyter lab notebooks/

api: setup
	$(VENV)/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

app: setup
	$(VENV)/bin/streamlit run app/app.py

docker:
	docker build -t heart-predictor .
	docker run -p 8000:8000 heart-predictor

clean:
	rm -rf $(VENV)
