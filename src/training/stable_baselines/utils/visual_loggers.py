import logging
import tempfile
from collections import Counter
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
        if not histories:
            return None

        num_episodes = len(histories)
        max_steps = max(len(h) for h in histories)

        full_s = self._max_tile_size
        half_s = full_s // 2

        grid_rows = []

        def empty_tile(w, h):
            t = np.zeros((h, w, 3), dtype=np.uint8)
            t[:] = 32
            return t

        # Header Row: [Empty, A, B, C, ...]
        header_row = [empty_tile(half_s, half_s)]
        for i in range(num_episodes):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            text = chr(65 + (i % 26))
            color = (255, 255, 255) # White
            
            if meta.get("mdp", False):
                text += " MDP"
                color = (0, 255, 0) # Green
            header_row.append(self._create_text_tile(text, width=full_s, height=half_s, color=color))
        grid_rows.append(header_row)

        # Info Row: [Stats, Info(0), Info(1), ...]
        info_row = [self._create_text_tile("Stats", width=half_s, height=full_s)]
        for i in range(num_episodes):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            info_row.append(self._create_info_tile(meta))
        grid_rows.append(info_row)

        # Step Rows
        for j in range(max_steps):
            row = [self._create_text_tile(str(j), width=half_s, height=full_s)]
            for i in range(num_episodes):
                history = histories[i]
                meta = metadata[i] if metadata and i < len(metadata) else {}
                step_history = meta.get("step_history", [])

                if j < len(histories[i]):
                    img = histories[i][j]
                    tile = self._process_image_tile(img, j, len(histories[i]), meta, step_history)
                    row.append(tile)
                else:
                    row.append(empty_tile(full_s, full_s))
            grid_rows.append(row)

        return np.vstack([np.hstack(row) for row in grid_rows])

    def _process_image_tile(self, img, j, history_len, meta, step_history):
        if img is None:
            t = np.zeros((self._max_tile_size, self._max_tile_size, 3), dtype=np.uint8)
            t[:] = 32
            return t

        # Platz für Text reservieren (unten)
        text_height = 30
        available_height = self._max_tile_size - text_height

        # Resize auf Kachelgröße
        h, w = img.shape[:2]
        scale = min(available_height / h, self._max_tile_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Auf quadratischen Canvas zentrieren
        canvas = np.zeros((self._max_tile_size, self._max_tile_size, 3), dtype=np.uint8)
        canvas[:] = 32
        y_offset = (available_height - new_h) // 2
        x_offset = (self._max_tile_size - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Rahmen um das letzte Bild zeichnen
        if j == history_len - 1:
            success = meta.get("success", False)
            truncated = meta.get("truncated", False)

            # Success/Fail Rahmen (Grün/Rot)
            color = (0, 255, 0) if success else (0, 0, 255)
            cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + new_w - 1, y_offset + new_h - 1), color, 2)

            if truncated:
                # Truncated Rahmen (Orange gestrichelt)
                self._draw_dashed_rect(canvas, (x_offset, y_offset), (x_offset + new_w - 1, y_offset + new_h - 1),
                                       (0, 140, 255), 2)

        # Textinformationen rendern (Label und Rewards)
        if j > 0 and (j - 1) < len(step_history):
            step_info = step_history[j - 1]
            label = str(step_info.get("label", ""))
            rew = step_info.get("reward", 0.0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            color = (200, 200, 200)
            thickness = 1

            # Farben definieren (BGR)
            color_pos = (100, 255, 100)  # Lindgrün
            color_neg = (0, 140, 255)  # Orange

            cv2.putText(canvas, label, (5, available_height + 12), font, font_scale, color, thickness, cv2.LINE_AA)

            # Zeile zusammenbauen: "r: <val> s: <val>"
            y_pos = available_height + 25
            x_pos = 5

            # Reward
            cv2.putText(canvas, "r:", (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
            x_pos += cv2.getTextSize("r:", font, font_scale, thickness)[0][0]

            rew_str = f"{rew:.4f}"
            cv2.putText(canvas, rew_str, (x_pos, y_pos), font, font_scale, color_neg if rew < 0 else color_pos,
                        thickness, cv2.LINE_AA)
            x_pos += cv2.getTextSize(rew_str, font, font_scale, thickness)[0][0]

            # Score
            score = step_info.get("score", 0.0)
            initial_score = meta.get("initial_score")
            cv2.putText(canvas, " s:", (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
            x_pos += cv2.getTextSize(" s:", font, font_scale, thickness)[0][0]

            score_str = f"{score:.4f}"
            is_bad_score = score < initial_score if initial_score is not None else score < 0
            cv2.putText(canvas, score_str, (x_pos, y_pos), font, font_scale, color_neg if is_bad_score else color_pos,
                        thickness, cv2.LINE_AA)

        return canvas

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

        # Transformer Stats hinzufügen
        step_history = meta.get("step_history", [])
        if step_history:
            counts = Counter([s.get("label") for s in step_history if s.get("label")])
            # Die 2 häufigsten anzeigen
            for label, count in counts.most_common(2):
                lines.append(f"{label[:9]}: {count}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (255, 255, 255)
        thickness = 1
        line_spacing = 25

        # Vertikal zentrieren
        total_height = len(lines) * line_spacing
        y_start = (self._max_tile_size - total_height) // 2 + 10

        for idx, line in enumerate(lines):
            y = y_start + idx * line_spacing
            cv2.putText(canvas, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return canvas

    def _create_text_tile(self, text: str, width: int = None, height: int = None, color: tuple = (255, 255, 255)) -> np.ndarray:
        w = width if width is not None else self._max_tile_size
        h = height if height is not None else self._max_tile_size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = 32

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8 if len(text) > 2 else 1.0
        thickness = 2

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2

        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
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

            # OpenH264 Limit Check
            MAX_PIXELS = 3840 * 2160
            current_pixels = max_w * max_h
            scale_factor = 1.0
            if current_pixels > MAX_PIXELS:
                scale_factor = (MAX_PIXELS / current_pixels) ** 0.5
                logger.warning(f"Video dimensions {max_w}x{max_h} too large. Scaling down by {scale_factor:.2f}.")
                max_w = int(max_w * scale_factor)
                max_h = int(max_h * scale_factor)

            # Dimensionen müssen für viele Codecs (z.B. H.264) gerade sein
            if max_h % 2 != 0: max_h += 1
            if max_w % 2 != 0: max_w += 1

            # 2. Video Writer initialisieren
            with tempfile.TemporaryDirectory() as tmp_dir:
                out = None
                video_path = None

                # Versuche verschiedene Codecs für maximale Kompatibilität
                # 1. avc1 (H.264) -> .mp4
                # 2. h264 (H.264) -> .mp4
                # 3. XVID (MPEG-4) -> .avi
                # 4. MJPG -> .avi
                attempts = [('avc1', '.mp4'), ('h264', '.mp4'), ('XVID', '.avi'), ('mp4v', '.avi'), ('MJPG', '.avi')]

                for codec, ext in attempts:
                    try:
                        current_path = Path(tmp_dir) / f"evaluation_timelapse{ext}"
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        temp_out = cv2.VideoWriter(str(current_path), fourcc, fps, (max_w, max_h))

                        if temp_out.isOpened():
                            out = temp_out
                            video_path = current_path
                            logger.info(f"VideoWriter initialized with codec '{codec}'")
                            break
                        else:
                            logger.warning(f"VideoWriter failed to open with codec '{codec}'")
                            if codec in ['avc1', 'h264']:
                                logger.info("Hint: Missing 'openh264-1.8.0-win64.dll'? Falling back to other codecs.")
                    except Exception as e:
                        logger.warning(f"Exception initializing VideoWriter with codec '{codec}': {e}")

                if out is None or not out.isOpened():
                    logger.error("Failed to open VideoWriter with any codec. Video generation skipped.")
                    return

                for img in images:
                    if scale_factor < 1.0:
                        h_orig, w_orig = img.shape[:2]
                        img = cv2.resize(img, (int(w_orig * scale_factor), int(h_orig * scale_factor)), interpolation=cv2.INTER_AREA)

                    h, w = img.shape[:2]
                    # Wenn Bild kleiner als Max, mit Schwarz auffüllen (zentrieren)
                    if h < max_h or w < max_w:
                        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                        # Links-Oben Ausrichtung, um Springen bei variabler Höhe zu vermeiden
                        y_off = 0
                        x_off = 0
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


class VisualStatisticsLogger:
    """
    Erstellt statistische Visualisierungen (z.B. Balkendiagramme) mittels OpenCV
    und loggt diese als Artefakte nach MLflow.
    Vermeidet Matplotlib-Abhängigkeiten für Thread-Safety und Performance.
    """

    def log_transformer_distribution(self, usage_counts: dict[str, int], step: int, prefix: str = "train"):
        if not usage_counts:
            return

        try:
            # Sortieren nach Häufigkeit
            sorted_items = sorted(usage_counts.items(), key=lambda x: x[1], reverse=True)
            labels = [k for k, v in sorted_items]
            values = [v for k, v in sorted_items]
            
            if not values:
                return

            # Canvas Konfiguration
            height_per_bar = 40
            margin_top = 40
            margin_bottom = 20
            margin_left = 250  # Platz für Text
            margin_right = 100 # Platz für Zahlen
            bar_height = 25
            
            img_h = margin_top + margin_bottom + (len(labels) * height_per_bar)
            img_w = 800
            
            # Weißer Hintergrund
            canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
            
            # Titel
            cv2.putText(canvas, f"Transformer Usage ({prefix} @ step {step})", (20, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)

            max_val = max(values)
            total = sum(values)
            
            # Zeichenbereich für Balken
            chart_width = img_w - margin_left - margin_right

            for i, (label, count) in enumerate(zip(labels, values)):
                y_pos = margin_top + (i * height_per_bar)
                
                # 1. Label Text (linksbündig im Margin)
                # Text kürzen falls zu lang
                display_label = label if len(label) < 30 else label[:27] + "..."
                cv2.putText(canvas, display_label, (10, y_pos + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # 2. Balken
                if max_val > 0:
                    bar_w = int((count / max_val) * chart_width)
                else:
                    bar_w = 0
                
                # Farbe basierend auf Hash des Labels (damit sie konsistent bleiben)
                # Einfacher Trick für "zufällige" aber stabile Pastellfarben
                h_val = hash(label)
                b = (h_val & 0xFF)
                g = ((h_val >> 8) & 0xFF)
                r = ((h_val >> 16) & 0xFF)
                # Abdunkeln/Aufhellen für Pastell-Look
                color = (int(b/2 + 100), int(g/2 + 100), int(r/2 + 100))

                cv2.rectangle(canvas, (margin_left, y_pos + 5), (margin_left + bar_w, y_pos + 5 + bar_height), color, -1)
                cv2.rectangle(canvas, (margin_left, y_pos + 5), (margin_left + bar_w, y_pos + 5 + bar_height), (200, 200, 200), 1)

                # 3. Wert und Prozent (rechts neben Balken)
                percent = (count / total) * 100 if total > 0 else 0
                text_val = f"{count} ({percent:.1f}%)"
                cv2.putText(canvas, text_val, (margin_left + bar_w + 10, y_pos + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Speichern und Hochladen
            with tempfile.TemporaryDirectory() as tmp_dir:
                filename = f"dist_{prefix}_{step:07d}.jpg"
                filepath = Path(tmp_dir) / filename
                cv2.imwrite(str(filepath), canvas)
                mlflow.log_artifact(str(filepath), artifact_path="transformer_stats")

        except Exception as e:
            logger.error(f"Failed to log visual statistics: {e}", exc_info=True)
