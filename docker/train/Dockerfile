FROM python:3.13-slim

# Arbeitsverzeichnis
WORKDIR /app

# Wenn keine GUI-Elemente von OpenCV-benötigt werden z.B. cv2.imshow()
# Minimale Abhängigkeiten installieren
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgomp1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copies the requirements.txt directoly form the root folder.
# ../../requirements.txt does not work because the context is set to the train directory
# and it is not possible to reference a file relative outside the context.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Default command: interactive shell for running experiments manually
# Can be overridden to run specific training scripts
CMD ["/bin/bash"]