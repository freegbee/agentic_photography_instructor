from typing import Optional, Tuple

from pydantic import BaseModel


class ScoringRequestPayloadV1(BaseModel):
    """Payload für Scoring-Anfragen in der Version V1"""
    filename: Optional[str] = None
    b64: Optional[str] = None


class ScoringResponsePayloadV1(BaseModel):
    """Payload für Scoring-Antworten in der Version V1"""
    filename: Optional[str] = None
    score: Optional[float] = None
    message: Optional[str] = None


class ScoringNdarrayRequestPayloadV1(BaseModel):
    """Metadaten-Payload für das Übertragen eines NumPy-ndarrays.

    Das Array selbst wird effizient als Binärdatei im Multipart-Upload (npy/npz)
    im Feld `array_npy` gesendet. Dieses Model beschreibt die optionalen
    Metadaten, die mitgeschickt werden können.
    """
    filename: Optional[str] = None
    # Encoding beschreibt das Format der Binärdatei, z.B. 'npy' oder 'npz'
    encoding: Optional[str] = "npy"
    # Optional: dtype und shape können zur Validierung mitgeschickt werden
    dtype: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None
