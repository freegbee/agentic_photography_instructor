FROM python:3.11-slim

# Install minimal system deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gcc build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy source

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV EXPORTER_PORT=8000

ENTRYPOINT ["python", "-u", "metrics_exporter/exporter.py"]

