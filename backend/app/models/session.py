from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .base import BaseModel


class ECGSession(BaseModel):
    __tablename__ = "ecg_sessions"
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    session_id = Column(String(100), unique=True, nullable=False)
    duration = Column(Float)  # seconds
    sample_rate = Column(Integer, default=360)
    data_points = Column(Integer, default=0)
    average_bpm = Column(Float)
    min_bpm = Column(Float)
    max_bpm = Column(Float)
    anomaly_count = Column(Integer, default=0)
    metadata = Column(JSON)  # Additional session data
    
    # Relationships
    patient = relationship("Patient", back_populates="sessions")
    anomalies = relationship("AnomalyLog", back_populates="session")
