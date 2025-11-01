#!/bin/sh
set -eu
# Standardwerte, falls die Env-Variablen nicht gesetzt sind
: "${JUROR_PORT:=5010}"
: "${MLFLOW_PORT:=5000}"
# Ersetze Variablen in der Template
envsubst '\$JUROR_PORT \$MLFLOW_PORT' < /etc/prometheus/prometheus.yml.template > /etc/prometheus/prometheus.yml
# Start Prometheus
exec /bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus