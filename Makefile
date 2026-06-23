# Halcyon Credit — developer task runner.
# Usage: `make <target>`. Run `make help` for the list.

.DEFAULT_GOAL := help
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help setup install lint format type test eval run clean ci

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install all dependencies
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo "✅ Environment ready. Copy .env.example to .env and add your keys."

install: ## Install/refresh dependencies into existing venv
	$(BIN)/pip install -r requirements.txt

lint: ## Lint + type-check (ruff + mypy)
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/mypy .

format: ## Auto-format the codebase (ruff)
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

type: ## Type-check only
	$(BIN)/mypy .

test: ## Run unit + integration tests
	$(BIN)/pytest

eval: ## Run the evaluation harness (golden set → RAGAS + fairness + baseline)  [Sprint 3]
	$(BIN)/python -m eval.regression_gate

run: ## Start the FastAPI service locally  [Sprint 2]
	$(BIN)/uvicorn api.main:app --reload --port 8000

ci: lint test ## What CI runs (lint + tests)

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ build dist *.egg-info
