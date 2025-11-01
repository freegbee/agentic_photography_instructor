from typing import Optional

from pydantic import BaseModel

class ScoringResponsePayload(BaseModel):
    filename: Optional[str] = None
    score: Optional[float] = None
    message: Optional[str] = None