#!/bin/bash
set -euo pipefail
# Health Check Script
API_URL=${API_URL:-"http://localhost:8000"}
MAX_RETRIES=30
RETRY_INTERVAL=5
echo "Checking API health at ${API_URL}..."
for i in $(seq 1 ${MAX_RETRIES}); do
    if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
        echo "API is healthy"
        if curl -sf "${API_URL}/ready" > /dev/null 2>&1; then
            echo "API is ready"
            exit 0
        else
            echo "API not ready (model not loaded?)"
        fi
    else
        echo "Attempt ${i}/${MAX_RETRIES}: API not responding..."
    fi
    sleep ${RETRY_INTERVAL}
done
echo "API health check failed after ${MAX_RETRIES} retries"
exit 1
