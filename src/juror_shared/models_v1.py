from typing import Optional

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
