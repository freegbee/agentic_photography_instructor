FROM python:3.13-slim

# ARG für Build-Zeit definieren
ARG MLFLOW_PORT

# Umgebungsvariable für Laufzeit setzen
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_PORT=${MLFLOW_PORT}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      openssh-client \
      ca-certificates \
      libglib2.0-0 \
      libgl1 \
      libgomp1 \
      curl && \
    rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /mlflow

COPY requirements.txt /mlflow/requirements.txt
COPY digital-ocean-postgres-ca-certificate.crt /usr/local/share/ca-certificates/certificate.crt

RUN update-ca-certificates

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# MLflow-Port
EXPOSE $MLFLOW_PORT