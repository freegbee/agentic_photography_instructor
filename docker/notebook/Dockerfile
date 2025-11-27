FROM python:3.13-slim

# ARG für Build-Zeit definieren
ARG NOTEBOOK_PORT

# Environment (runtime overridable via docker-compose/.env)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NOTEBOOK_PORT=${NOTEBOOK_PORT}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      git \
      tini \
      libglib2.0-0 \
      libgl1 \
      libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /notebook

# WICHTIG: explizit die Notebook-spezifischen Requirements kopieren
# (vermeidet, dass versehentlich die Root-requirements installiert werden)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install jupyterlab ipykernel jupytext && \
    pip install -r requirements.txt

EXPOSE $NOTEBOOK_PORT
