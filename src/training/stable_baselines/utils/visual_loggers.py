import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import mlflow
import numpy as np

logger = logging.getLogger(__name__)


class VisualSnapshotLogger:
    """
    Kapselt die Logik zur Erstellung und zum Logging des visuellen Mosaiks (Snapshot).
    """

    def __init__(self, max_tile_size: int = 150):
        self._max_tile_size = max_tile_size

    def log_summary(self, histories: List[List[np.ndarray]], evaluation_idx: int, save_dir: Path = None,
                    metadata: Optional[List[dict]] = None) -> Optional[
        Path]:
        """
        Erstellt ein Mosaik aus den Bildverläufen und lädt es als MLflow Artefakt hoch.
        :param histories: Liste von np.array-listen mit den Bildinformationen nach dem jeweiligen step
        :param evaluation_idx: Index der Evaluation
        :param save_dir: Wenn angegeben, wird das Bild dort gespeichert. Sonst in einem temp dir.
        :return: Pfad zur gespeicherten Datei
        """
        try:
            mosaic = self._generate_mosaic(histories, metadata)
            if mosaic is None:
                return None

            # Wir nutzen hier Logik, die beides abdeckt.
            # Wenn save_dir None ist, erstellen wir einen temporären Context nur für den Upload
            if save_dir is None:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    return self._save_and_log(mosaic, Path(tmp_dir), evaluation_idx)
            else:
                return self._save_and_log(mosaic, save_dir, evaluation_idx)

        except Exception as e:
            logger.error(f"Failed to generate or log visual summary: {e}", exc_info=True)
            return None

    def _save_and_log(self, mosaic: np.ndarray, directory: Path, evaluation_idx: int) -> Path:
        filename = f"eval_mosaic_{evaluation_idx:03d}.jpg"
        filepath = directory / filename
        # JPG-Qualität (0-100). Höher ist besser. 85 ist ein guter Standardwert.
        cv2.imwrite(str(filepath), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 85])
        mlflow.log_artifact(str(filepath), artifact_path="evaluation_images")
        logger.info(f"Logged evaluation mosaic: {filename}")
        return filepath

    def _generate_mosaic(self, histories: List[List[np.ndarray]], metadata: Optional[List[dict]] = None) -> Optional[np.ndarray]:
        rows = []
        for i, history in enumerate(histories):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            row_images = []
            
            # Info-Kachel erstellen und als erstes Element (links) hinzufügen
            row_images.append(self._create_info_tile(meta))
            
            for j, img in enumerate(history):
                if img is None:
                    continue

                # Resize auf Kachelgröße (falls noch nicht geschehen oder abweichend)
                h, w = img.shape[:2]
                scale = min(self._max_tile_size / h, self._max_tile_size / w)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Auf quadratischen Canvas zentrieren (für sauberes Grid)
                canvas = np.zeros((self._max_tile_size, self._max_tile_size, 3), dtype=np.uint8)
                canvas[:] = 32
                y_offset = (self._max_tile_size - new_h) // 2
                x_offset = (self._max_tile_size - new_w) // 2
                canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

                # Rahmen um das letzte Bild zeichnen
                if j == len(history) - 1:
                    success = meta.get("success", False)
                    truncated = meta.get("truncated", False)
                    
                    # Success/Fail Rahmen (Grün/Rot)
                    color = (0, 255, 0) if success else (0, 0, 255)
                    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + new_w - 1, y_offset + new_h - 1), color, 2)
                    
                    if truncated:
                        # Truncated Rahmen (Orange gestrichelt)
                        self._draw_dashed_rect(canvas, (x_offset, y_offset), (x_offset + new_w - 1, y_offset + new_h - 1), (0, 140, 255), 2)

                row_images.append(canvas)

            if row_images:
                rows.append(np.hstack(row_images))

        if not rows:
            return None

        # Zeilen auf gleiche Breite bringen
        max_width = max(row.shape[1] for row in rows)
        padded_rows = []
        for row in rows:
            h, w = row.shape[:2]
            if w < max_width:
                padding = np.zeros((h, max_width - w, 3), dtype=np.uint8)
                padding[:] = 32
                row = np.hstack([row, padding])
            padded_rows.append(row)

        return np.vstack(padded_rows)

    def _create_info_tile(self, meta: dict) -> np.ndarray:
        canvas = np.zeros((self._max_tile_size, self._max_tile_size, 3), dtype=np.uint8)
        canvas[:] = 32

        initial = meta.get("initial_score")
        final = meta.get("score")

        lines = []
        if initial is not None:
            lines.append(f"i: {initial:.4f}")
        else:
            lines.append("i: N/A")

        if final is not None:
            lines.append(f"f: {final:.4f}")
        else:
            lines.append("f: N/A")

        if initial is not None and final is not None:
            delta = final - initial
            lines.append(f"d: {delta:+.4f}")
        else:
            lines.append("d: N/A")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (255, 255, 255)
        thickness = 1
        line_spacing = 25

        # Vertikal zentrieren
        total_height = len(lines) * line_spacing
        y_start = (self._max_tile_size - total_height) // 2 + 15

        for idx, line in enumerate(lines):
            y = y_start + idx * line_spacing
            cv2.putText(canvas, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return canvas

    def _draw_dashed_rect(self, img, pt1, pt2, color, thickness=1, dash_len=5):
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top and Bottom
        for x in range(x1, x2, dash_len * 2):
            cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
            
        # Left and Right
        for y in range(y1, y2, dash_len * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)


class VisualTrainingLogger:
    """
    Kapselt die Logik zur Erstellung und zum Logging von Trainings-Artefakten wie Videos.
    """

    def log_video(self, image_paths: List[Path], fps: int = 2):
        if not image_paths:
            return

        try:
            logger.info(f"Generating evaluation video from {len(image_paths)} frames...")

            # 1. Maximale Dimensionen ermitteln (da VideoWriter fixe Größe braucht)
            max_h, max_w = 0, 0
            images = []
            for p in image_paths:
                img = cv2.imread(str(p))
                if img is not None:
                    h, w = img.shape[:2]
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)
                    images.append(img)

            if not images:
                return

            # 2. Video Writer initialisieren
            with tempfile.TemporaryDirectory() as tmp_dir:
                video_path = Path(tmp_dir) / "evaluation_timelapse.mp4"
                # mp4v ist weit verbreitet, alternativ 'avc1'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(video_path), fourcc, fps, (max_w, max_h))

                for img in images:
                    h, w = img.shape[:2]
                    # Wenn Bild kleiner als Max, mit Schwarz auffüllen (zentrieren)
                    if h < max_h or w < max_w:
                        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                        y_off = (max_h - h) // 2
                        x_off = (max_w - w) // 2
                        canvas[y_off:y_off + h, x_off:x_off + w] = img
                        out.write(canvas)
                    else:
                        out.write(img)

                out.release()

                # 3. Hochladen
                mlflow.log_artifact(str(video_path), artifact_path="evaluation_video")
                logger.info(f"Logged evaluation video: {video_path.name}")

        except Exception as e:
            logger.error(f"Failed to generate evaluation video: {e}", exc_info=True)
