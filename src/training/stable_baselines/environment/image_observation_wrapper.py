from typing import Tuple

import numpy as np
from cv2 import resize
from gymnasium import ObservationWrapper, spaces
from numpy import ndarray


class ImageObservationWrapper(ObservationWrapper):
    """
        ObservationWrapper, der rohe HWC-Bilder (np.uint8 oder float) annimmt,
        auf image_max_size skaliert, normalisiert und in CHW float32 zurückgibt.
        """

    def __init__(self, env, image_max_size: Tuple[int, int]):
        super().__init__(env)
        self.image_max_size = image_max_size  # (h, w)
        h, w = image_max_size
        # Output: CHW, Werte in [0,1]
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3, h, w), dtype=np.float32)

    def observation(self, obs) -> ndarray:
        """
        obs: np.ndarray in HWC (Height, Width, Channels), BGR (cv2) or RGB, dtype uint8 oder float.
        Rückgabe: np.ndarray CHW float32 im Bereich [0,1].
        """
        img = obs

        if not isinstance(img, np.ndarray):
            raise TypeError("Observation is not a numpy array")

        # Normierung
        if img.dtype == np.uint8:
            img_proc = img.astype(np.float32) / 255.0
        else:
            img_proc = img.astype(np.float32)
            # Falls Werte in 0..255 vorliegen
            if img_proc.max() > 1.5:
                img_proc = img_proc / 255.0

        # Resize auf (width, height) für cv2.resize
        target_size = self.image_max_size[::-1]
        img_resized = resize(img_proc, target_size, interpolation=1)

        # Falls 4 Kanäle (RGBA), nur RGB/BGR behalten
        if img_resized.ndim == 3 and img_resized.shape[2] == 4:
            img_resized = img_resized[:, :, :3]

        # Konvertierung von BGR zu RGB für ResNet (erwartet RGB)
        if img_resized.ndim == 3 and img_resized.shape[2] == 3:
            img_resized = img_resized[..., ::-1]

        img_clipped = np.clip(img_resized, 0.0, 1.0).astype(np.float32)

        # HWC -> CHW
        return np.transpose(img_clipped, (2, 0, 1))
