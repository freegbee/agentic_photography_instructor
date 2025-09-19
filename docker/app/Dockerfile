FROM python:3.13-slim

# ARG für Build-Zeit definieren
ARG MLFLOW_PORT

# Umgebungsvariable für Laufzeit setzen
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_PORT=${MLFLOW_PORT}

# Arbeitsverzeichnis
WORKDIR /app

#COPY ../../requirements.txt /tmp/requirements.txt
#RUN python -m pip install --upgrade pip && \
#    pip install -r /tmp/requirements.txt