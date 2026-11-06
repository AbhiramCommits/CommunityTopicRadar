PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTHON_VERSION ?= python3.11

export PYTHONPATH := src

.PHONY: setup ingest analyze brief export dashboard test lint all

.NOTPARALLEL:

setup:
	$(PYTHON_VERSION) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

ingest:
	$(PYTHON) -m ctr.cli ingest --offline

analyze:
	$(PYTHON) -m ctr.cli analyze

brief:
	$(PYTHON) -m ctr.cli brief

export:
	$(PYTHON) -m ctr.cli export

dashboard:
	$(PYTHON) -m streamlit run src/ctr/dashboard.py

test:
	$(PYTHON) -m pytest --cov=src/ctr --cov-report=term-missing --cov-fail-under=85

lint:
	$(PYTHON) -m ruff check .

all: ingest analyze brief export
