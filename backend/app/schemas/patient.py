from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date


class PatientBase(BaseModel):
    medical_id: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: Optional[str] = None
    height: Optional[int] = None
    weight: Optional[int] = None
    blood_type: Optional[str] = None
    allergies: Optional[str] = None
    medical_history: Optional[str] = None


class PatientCreate(PatientBase):
    pass


class PatientRead(PatientBase):
    id: int
    created_at: Optional[datetime]

    class Config:
        orm_mode = True
