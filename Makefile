.PHONY: help install data train test lint format serve clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

data:  ## Generate synthetic data
	python run.py data --config configs/default.yaml

train:  ## Run the full training pipeline
	python run.py train --config configs/default.yaml

test:  ## Run the test suite
	pytest -v

serve:  ## Start the API server
	uvicorn mlops.serving.app:app --host 0.0.0.0 --port 8000 --reload

lint:  ## Lint
	ruff check mlops tests run.py || true

format:  ## Format code
	ruff format mlops tests run.py || true

clean:  ## Remove generated artifacts
	rm -rf mlruns data/*.csv *.log metrics.json .pytest_cache __pycache__ .ruff_cache
