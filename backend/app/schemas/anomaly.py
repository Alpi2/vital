from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class AnomalyBase(BaseModel):
    anomaly_type: str = Field(..., description="Type of anomaly detected")
    severity: str = Field(..., pattern="^(low|medium|high)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    duration: Optional[float] = None
    bpm_at_detection: Optional[float] = None
    details: Optional[str] = None

class AnomalyCreate(AnomalyBase):
    patient_id: int
    session_id: Optional[int] = None

class AnomalyResponse(AnomalyBase):
    id: int
    patient_id: int
    session_id: Optional[int]
    created_at: datetime
    resolved: bool
    
    class Config:
        from_attributes = True

class AnomalyStats(BaseModel):
    total_count: int
    by_type: dict[str, int]
    by_severity: dict[str, int]
    today_count: int
    unresolved_count: int
