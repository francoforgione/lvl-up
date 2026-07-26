.PHONY: install dev-up dev-down init-db ingest index ground-truth eval-retrieval eval-rag app up down test

VENV=.venv
PYTHON=$(VENV)/bin/python

install:
	python3.11 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

dev-up:
	docker compose -f docker-compose.dev.yml up -d

dev-down:
	docker compose -f docker-compose.dev.yml down

init-db:
	$(PYTHON) -c "from lvlup.monitoring.db import init_schema; init_schema()"

ingest:
	$(PYTHON) -m lvlup.ingestion.pipeline

index:
	$(PYTHON) scripts/run_pipeline.py index

ground-truth:
	$(PYTHON) -m lvlup.evaluation.generate_ground_truth

eval-retrieval:
	$(PYTHON) -m lvlup.evaluation.retrieval_eval

eval-rag:
	$(PYTHON) -m lvlup.evaluation.rag_eval

app:
	$(VENV)/bin/streamlit run app/streamlit_app.py

up:
	docker compose up --build -d

down:
	docker compose down

test:
	$(PYTHON) -m pytest tests/
