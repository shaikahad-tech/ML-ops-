#!/bin/bash
set -euo pipefail
# Blue-Green Deployment Script
NEW_VERSION=${1:-latest}
NAMESPACE=${2:-default}
SERVICE_NAME=mlops-api
echo "Starting blue-green deployment for version ${NEW_VERSION}..."
CURRENT=$(kubectl get deployment -n ${NAMESPACE} \
    -l app=${SERVICE_NAME} -o jsonpath="{.items[0].metadata.name}" 2>/dev/null || echo "")
if [ "$CURRENT" = "${SERVICE_NAME}-green" ]; then
    NEW_COLOR="blue"
    OLD_COLOR="green"
else
    NEW_COLOR="green"
    OLD_COLOR="blue"
fi
echo "Current: ${OLD_COLOR}, Deploying: ${NEW_COLOR}"
kubectl apply -f deploy/kubernetes/api-deployment.yaml -n ${NAMESPACE}
echo "Waiting for ${NEW_COLOR} deployment..."
kubectl rollout status deployment/${SERVICE_NAME}-${NEW_COLOR} -n ${NAMESPACE}
echo "Switching traffic to ${NEW_COLOR}..."
kubectl patch svc ${SERVICE_NAME} -n ${NAMESPACE} \
    -p "{\"spec\":{\"selector\":{\"version\":\"${NEW_COLOR}\"}}}"
echo "Waiting 60s before cleaning up ${OLD_COLOR}..."
sleep 60
kubectl delete deployment ${SERVICE_NAME}-${OLD_COLOR} -n ${NAMESPACE} || true
echo "Blue-green deployment complete"
