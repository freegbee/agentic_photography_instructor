from typing import Optional

from pydantic import BaseModel

class ScoringRequestPayload(BaseModel):
    filename: Optional[str] = None
    b64: Optional[str] = None