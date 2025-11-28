FROM python:3.13-slim


# Arbeitsverzeichnis
WORKDIR /app


# Notwendige Bibliotheken für OpenCV installieren
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      curl \
      openssh-client \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copies the requirements.txt directoly form the root folder.
# ../../requirements.txt does not work because the context is set to the image-acquisition-service directory
# and it is not possible to reference a file releative outside the context.
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt