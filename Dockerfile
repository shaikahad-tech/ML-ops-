# Build the API image:
#   docker build -t churn-api .
# Run it:
#   docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=file:./mlruns -v ${PWD}/mlruns:/app/mlruns churn-api
#
# Train inside a container first so a registered model exists:
#   docker run --rm -v ${PWD}:/app churn-api python run.py train --config configs/default.yaml

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project.
COPY . .

EXPOSE 8000

# Train by default? No — serve. Override the command to train:
#   docker run --rm churn-api python run.py train --config configs/default.yaml
CMD ["uvicorn", "mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
