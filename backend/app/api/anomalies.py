from fastapi import APIRouter
from typing import List
from ..schemas.anomaly import AnomalyResponse, AnomalyCreate
from datetime import datetime

router = APIRouter()


@router.get("/", response_model=List[AnomalyResponse])
async def list_anomalies():
    return []


@router.post("/", response_model=AnomalyResponse)
async def create_anomaly(payload: AnomalyCreate):
    now = datetime.utcnow()
    return {
        "id": 1,
        **payload.dict(),
        "timestamp": now,
        "created_at": now,
        "resolved": False,
    }
