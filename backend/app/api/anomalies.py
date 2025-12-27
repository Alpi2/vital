from fastapi import APIRouter
from typing import List
from ..schemas.anomaly import AnomalyRead, AnomalyCreate

router = APIRouter()


@router.get("/", response_model=List[AnomalyRead])
async def list_anomalies():
    return []


@router.post("/", response_model=AnomalyRead)
async def create_anomaly(payload: AnomalyCreate):
    return {"id": 1, **payload.dict(), "timestamp": None}
