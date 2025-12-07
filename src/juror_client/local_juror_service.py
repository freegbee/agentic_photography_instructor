import os
import logging
from typing import Optional, Union, Dict, Any

import numpy as np
from PIL import Image

from juror_client.juror_service import JurorService
from juror_shared.models_v1 import ScoringResponsePayloadV1

# Importiere die lokale Juror-Klasse (aus dem Server-Paket)
from juror_server.juror.Juror import Juror

logger = logging.getLogger(__name__)


class LocalJurorService(JurorService):
    """Lokale JurorService-Implementierung, die `Juror` direkt verwendet.

    Nutzbar z.B. in Tests oder wenn kein HTTP-Server gewünscht ist.
    """

    def __init__(self, juror: Optional[Juror] = None) -> None:
        self._juror = juror or Juror()

    def _prepare_array(self, array: np.ndarray) -> np.ndarray:
        """Sichere Normalisierung/Formatierung des Arrays für Juror.inference.

        Erwartet ein HxWxC RGB-Array mit Werten in [0,255].
        - Falls Channels-first (C,H,W) wird transponiert.
        - Falls Graustufen (H,W) wird zu RGB gestapelt.
        - Falls Float in [0,1] -> skaliert auf [0,255].
        - Liefert dtype uint8.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array muss ein numpy.ndarray sein")

        arr = array
        # Channels-first -> channels-last
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] <= 3 and arr.shape[0] != arr.shape[2]:
            # Annahme: (C,H,W)
            arr = np.transpose(arr, (1, 2, 0))

        # Graustufen -> RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim != 3 or arr.shape[2] not in (1, 3):
            raise ValueError("Array muss Form HxW oder HxWxC (C=1 oder 3) haben")

        # Wenn single-channel nach obigen checks -> expandieren
        if arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=2)

        # Falls Float in [0,1]
        if np.issubdtype(arr.dtype, np.floating):
            maxv = float(np.nanmax(arr))
            if maxv <= 1.001:
                arr = (arr * 255.0)
        # Clip und cast
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def score_image(self, image_path: str) -> Union[ScoringResponsePayloadV1, Dict[str, Any], str]:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img)
        score = float(self._juror.inference(arr))
        # Versuche, das Modellobjekt zu erzeugen, fallback auf dict
        try:
            return ScoringResponsePayloadV1(score=score, filename=os.path.basename(image_path))
        except Exception:
            logger.debug("Could not build ScoringResponsePayloadV1, returning dict fallback")
            return {"score": score, "filename": os.path.basename(image_path)}

    def score_ndarray(self, array: np.ndarray, filename: Optional[str] = None, encoding: str = "npy") -> Union[ScoringResponsePayloadV1, Dict[str, Any], str]:
        if not isinstance(array, np.ndarray):
            raise TypeError("array muss ein numpy.ndarray sein")

        arr = self._prepare_array(array)
        score = float(self._juror.inference(arr))
        try:
            return ScoringResponsePayloadV1(score=score, filename=(filename or "array"))
        except Exception:
            logger.debug("Could not build ScoringResponsePayloadV1, returning dict fallback")
            return {"score": score, "filename": (filename or "array")}

    def close(self) -> None:
        # Lokale Juror-Instanz hat keine speziellen Ressourcen freizugeben
        return None