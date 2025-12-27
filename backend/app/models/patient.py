from sqlalchemy import Column, String, Integer, Date, Text
from sqlalchemy.orm import relationship
from .base import BaseModel


class Patient(BaseModel):
    __tablename__ = "patients"
    
    medical_id = Column(String(50), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(String(10))
    height = Column(Integer)  # cm
    weight = Column(Integer)  # kg
    blood_type = Column(String(5))
    allergies = Column(Text)
    medical_history = Column(Text)
    
    # Relationships
    sessions = relationship("ECGSession", back_populates="patient")
    anomalies = relationship("AnomalyLog", back_populates="patient")
