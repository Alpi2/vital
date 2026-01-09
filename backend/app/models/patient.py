from sqlalchemy import Column, String, Integer, Date, Text, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class Patient(BaseModel):
    __tablename__ = "patients"
    
    medical_id = Column(String(50), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False, index=True)
    last_name = Column(String(100), nullable=False, index=True)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(String(10))
    height = Column(Integer)  # cm
    weight = Column(Integer)  # kg
    blood_type = Column(String(5))
    allergies = Column(Text)
    medical_history = Column(Text)
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_name', 'last_name', 'first_name'),
        Index('idx_dob_name', 'date_of_birth', 'last_name'),
    )
    
    # Relationships
    sessions = relationship("ECGSession", back_populates="patient", cascade="all, delete-orphan")
    anomalies = relationship("AnomalyLog", back_populates="patient", cascade="all, delete-orphan")
