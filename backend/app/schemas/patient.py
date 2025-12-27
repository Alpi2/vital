from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class PatientBase(BaseModel):
    external_id: str
    name: str
    age: Optional[int] = None


class PatientCreate(PatientBase):
    pass


class PatientRead(PatientBase):
    id: int
    created_at: Optional[datetime]

    class Config:
        orm_mode = True
