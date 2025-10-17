FROM python:3.13-slim

# Arbeitsverzeichnis
WORKDIR /app

# Notwendige Bibliotheken für OpenCV installieren
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Wenn keine GUI-Elemente von OpenCV-benötigt werden z.B. cv2.imshow()
# Minimale Abhängigkeiten installieren
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#      libgomp1 \
#      libglib2.0-0 \
#    && rm -rf /var/lib/apt/lists/*

# Copies the requirements.txt directoly form the root folder.
# ../../requirements.txt does not work because the context is set to the data-service directory
# and it is not possible to reference a file releative outside the context.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Python-Modul-Notation mit -m test
# Das -m Flag teilt Python mit, dass es ein Modul ausführen soll, nicht eine Datei
CMD ["python", "/app/src/image_aquisition/DockerDownloadStarter.py"]