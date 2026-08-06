#!/bin/bash
set -euo pipefail
# MLOps Platform Release Script
VERSION=$(python -c "import mlops; print(mlops.__version__)")
echo "Releasing MLOps Platform v${VERSION}..."
echo "Running tests..."
pytest -v --tb=short
echo "Running lint..."
ruff check src tests
ruff format --check src tests
echo "Building Docker images..."
docker build -t mlops-api:${VERSION} -f docker/Dockerfile.api .
docker tag mlops-api:${VERSION} mlops-api:latest
echo "Running training pipeline..."
python -m mlops.cli pipeline --config configs/production.yaml
echo "Evaluating model..."
python -m mlops.cli evaluate --model-name churn-classifier --version latest
echo "Deploying to staging..."
python -m mlops.cli deploy --model-name churn-classifier --version latest --env staging
echo "Creating git tag..."
git tag -a "v${VERSION}" -m "Release v${VERSION}"
echo "Release v${VERSION} complete. Push with: git push origin --tags"
