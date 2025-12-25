import logging
from pathlib import Path
from typing import Any

import cv2
import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

logger = logging.getLogger(__name__)


class ImageRenderWrapper(gymnasium.Wrapper):
    metadata = {"render_modes": ["human", "rgb_array", "save"]}

    def __init__(self,
                 env,
                 render_mode: str = "imshow",  # | "save"
                 render_save_dir: Path = None, ):
        super(ImageRenderWrapper, self).__init__(env)
        self._render_mode: str = render_mode
        self._render_save_dir: Path = render_save_dir
        self._initial_image_to_render = None
        self._terminated_image_to_render = None

    def reset(self, **kwargs) -> Any:
        obs, info = self.env.reset(**kwargs)
        self._capture_current_image(obs, info, True)
        self._terminated_image_to_render = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            self._capture_current_image(obs, info, False)
            self.render()
        return obs, reward, terminated, truncated, info

    def _capture_current_image(self, obs, info, on_reset: bool):
        # Aktuelles Bild aus dem (gewrappten) env holen und intern speichern
        # Das bild könnte theoretisch auch in info oder obs sein. Hier wird angenommen, dass das env ein Attribut current_image hat
        img = getattr(self.env, "current_image", None)
        current_image_id = getattr(self.env, "current_image_id", None)
        logger.info(
            f"_capture_current_image: current_image_id: {current_image_id}, img is None: {img is None}, on_reset: {on_reset}")
        if on_reset:
            self._initial_image_to_render = np.array(img, copy=True) if img is not None else None
            self._current_image_id = current_image_id
        else:
            self._terminated_image_to_render = np.array(img, copy=True) if img is not None else None

    def close(self) -> None:
        self.env.close()
        try:
            plt.close("all")
        except Exception:
            pass

    def render(self, mode: str = None, step: int = None):
        mode = self._render_mode
        logger.info(f"render mode: {mode}, step: {step}, self._render_save_dir: {self._render_save_dir}")
        if self._initial_image_to_render is None or self._terminated_image_to_render is None:
            return None

        img_initial_rgb = self._preprocess_image_for_render(self._initial_image_to_render)
        img_terminated_rgb = self._preprocess_image_for_render(self._terminated_image_to_render)

        if mode == "rgb_array":
            # Gym erwartet üblicherweise HWC uint8 im Bereich [0,255]
            return (np.clip(img_terminated_rgb, 0.0, 1.0) * 255).astype(np.uint8)
        elif mode == "imshow":
            # matplotlib anzeigen (blockierendes Fenster vermeiden: use plt.pause)
            plt.figure("ImageTransformEnv")
            plt.axis("off")
            plt.imshow(img_terminated_rgb)
            plt.title(f"img_id={self._current_image_id} steps={self.step_count} score={self.current_score:.3f}")
            plt.pause(0.001)
            plt.draw()
        elif mode == "save":
            # als PNG speichern
            if not self._render_save_dir:
                return None
            # filename_initial = self._image_filename(True)
            # filename_terminated = self._image_filename(False)
            # os.makedirs(self.render_save_dir / str(self.reset_idx), exist_ok=True)
            # destination_path = self._render_save_dir / filename_terminated
            plt.imsave(self._render_save_dir / self._image_filename(True), img_initial_rgb)
            plt.imsave(self._render_save_dir / self._image_filename(False), img_terminated_rgb)
        else:
            # unbekannter Modus: stille Rückgabe
            return None
        return None

    def _preprocess_image_for_render(self, image_data: ndarray) -> ndarray:
        """Preprocess the image data for rendering: convert BGR to RGB"""
        return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    def _image_filename(self, on_reset: bool):
        return f"img_{self._current_image_id}_{"0" if on_reset else "1"}.png"
