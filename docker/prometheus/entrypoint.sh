#!/bin/sh
set -eu
# Standardwerte, falls die Env-Variablen nicht gesetzt sind
: "${IMAGE_ACQUISITION_PORT:=5005}"
: "${JUROR_PORT:=5010}"
: "${MLFLOW_PORT:=5000}"
: "${MONITORING_PROMETHEUS_PORT:=5020}"
# Ersetze Variablen in der Template
envsubst '\$JUROR_PORT \$IMAGE_ACQUISITION_PORT \$MLFLOW_PORT \$MONITORING_PROMETHEUS_PORT' < /etc/prometheus/prometheus.yml.template > /etc/prometheus/prometheus.yml
# Start Prometheus - nicht auf dem default port 9090, sondern auf dem konfigurierten Port
exec /bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.listen-address=0.0.0.0:${MONITORING_PROMETHEUS_PORT}