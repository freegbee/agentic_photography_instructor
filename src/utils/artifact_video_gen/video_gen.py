import argparse
import logging
import tempfile
from pathlib import Path
from typing import List

import cv2
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

# Logging Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_video_from_images(image_paths: List[Path], output_path: Path, fps: int = 2):
    """
    Erstellt ein Video aus einer Liste von Bildpfaden.
    Achtet auf gerade Dimensionen und zentriert Bilder, falls Größen variieren.
    """
    if not image_paths:
        logger.warning("No images provided for video generation.")
        return

    # 1. Maximale Dimensionen ermitteln
    max_h, max_w = 0, 0
    images = []

    logger.info(f"Reading {len(image_paths)} images...")
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            images.append(img)
        else:
            logger.warning(f"Failed to read image: {p}")

    if not images:
        logger.error("No valid images loaded.")
        return

    # OpenH264 Limit Check (max ca. 9.4MP). Wir nutzen 4K (8.3MP) als sicheres Limit.
    MAX_PIXELS = 3840 * 2160
    current_pixels = max_w * max_h
    scale_factor = 1.0

    if current_pixels > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / current_pixels) ** 0.5
        logger.warning(f"Dimensions {max_w}x{max_h} exceed codec limits. Scaling down by {scale_factor:.2f}.")
        max_w = int(max_w * scale_factor)
        max_h = int(max_h * scale_factor)

    # Dimensionen müssen für viele Codecs gerade sein
    if max_h % 2 != 0: max_h += 1
    if max_w % 2 != 0: max_w += 1

    logger.info(f"Video dimensions: {max_w}x{max_h}")

    # Codec Versuche:
    # 1. avc1 (H.264) -> .mp4 (Standard, hohe Kompatibilität)
    # 2. h264 (H.264) -> .mp4 (Alternative FourCC)
    # 3. XVID (MPEG-4) -> .avi (Sehr robust auf Windows)
    # 4. MJPG -> .avi (Fallback, große Dateien)
    attempts = [('avc1', '.mp4'), ('h264', '.mp4'), ('XVID', '.avi'), ('mp4v', '.avi'), ('MJPG', '.avi')]

    out = None
    final_output_path = output_path

    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    for codec, ext in attempts:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            # Pfad anpassen, falls Extension nicht zum Codec passt (z.B. Fallback auf .avi)
            current_output_path = output_path
            if output_path.suffix.lower() != ext:
                current_output_path = output_path.with_suffix(ext)

            temp_out = cv2.VideoWriter(str(current_output_path), fourcc, fps, (max_w, max_h))

            if temp_out.isOpened():
                out = temp_out
                final_output_path = current_output_path
                logger.info(f"VideoWriter initialized with codec '{codec}' to file '{final_output_path.name}'")
                break
            else:
                logger.warning(f"VideoWriter failed to open with codec '{codec}'")
                if codec in ['avc1', 'h264']:
                    logger.info("Hint: For H.264 (.mp4) support on Windows, download 'openh264-1.8.0-win64.dll' "
                                "and place it in your Python/Script directory.")
        except Exception as e:
            logger.warning(f"Exception initializing VideoWriter with codec '{codec}': {e}")

    if out is None or not out.isOpened():
        logger.error("Failed to open VideoWriter with any codec.")
        return

    # Frames schreiben
    for img in images:
        if scale_factor < 1.0:
            h_orig, w_orig = img.shape[:2]
            img = cv2.resize(img, (int(w_orig * scale_factor), int(h_orig * scale_factor)), interpolation=cv2.INTER_AREA)

        h, w = img.shape[:2]
        # Padding (Zentrierung) wenn Bild kleiner als Max
        if h < max_h or w < max_w:
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            y_off = 0
            x_off = 0
            canvas[y_off:y_off + h, x_off:x_off + w] = img
            out.write(canvas)
        else:
            out.write(img)

    out.release()
    logger.info(f"Video successfully created at: {final_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate video from MLflow evaluation artifacts.")
    parser.add_argument("--run-id", help="MLflow Run UUID")
    parser.add_argument("--output", default="evaluation_video.mp4", help="Output video file path")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")
    parser.add_argument("--artifact-path", default="evaluation_images", help="Artifact subfolder to download")
    parser.add_argument("--mlflow-uri", default=None, help="MLflow Tracking URI (optional)")

    args = parser.parse_args()

    # Interaktiver Modus, falls run-id fehlt
    if args.run_id is None:
        print("--- Interactive Mode ---")
        while not args.run_id:
            args.run_id = input("Please enter MLflow Run UUID: ").strip()

        val = input(f"Output video file path [{args.output}]: ").strip()
        if val:
            args.output = val

        val = input(f"Frames per second [{args.fps}]: ").strip()
        if val:
            try:
                args.fps = int(val)
            except ValueError:
                print(f"Invalid FPS. Using default: {args.fps}")

        val = input(f"Artifact subfolder to download [{args.artifact_path}]: ").strip()
        if val:
            args.artifact_path = val

        print("------------------------")

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    logger.info(f"Using MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    client = MlflowClient()

    logger.info(f"Connecting to MLflow and downloading artifacts for run {args.run_id}...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            local_path = client.download_artifacts(args.run_id, args.artifact_path, dst_path=tmp_dir)
        except Exception as e:
            logger.error(f"Failed to download artifacts: {e}")
            return

        image_dir = Path(local_path)
        # Bilder sammeln und sortieren (lexikographisch nach Dateiname)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ], key=lambda x: x.name)

        if not image_paths:
            logger.warning(f"No images found in artifact path '{args.artifact_path}'.")
            return

        logger.info(f"Found {len(image_paths)} images. Starting video generation...")

        output_path = Path(args.output).resolve()
        create_video_from_images(image_paths, output_path, args.fps)


if __name__ == "__main__":
    main()
