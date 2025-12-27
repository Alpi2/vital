from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ..schemas.patient import PatientRead, PatientCreate

router = APIRouter()


@router.get("/", response_model=List[PatientRead])
async def list_patients():
    # placeholder - integrate DB session later
    return []


@router.post("/", response_model=PatientRead)
async def create_patient(payload: PatientCreate):
    # placeholder - create and return
    return {"id": 1, **payload.dict(), "created_at": None}
