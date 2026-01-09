from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class ECGSession(BaseModel):
    __tablename__ = "ecg_sessions"
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    duration = Column(Float)  # seconds
    sample_rate = Column(Integer, default=360)
    data_points = Column(Integer, default=0)
    average_bpm = Column(Float)
    min_bpm = Column(Float)
    max_bpm = Column(Float)
    anomaly_count = Column(Integer, default=0)
    metadata = Column(JSON)  # Additional session data
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_patient_created', 'patient_id', 'created_at'),
        Index('idx_patient_duration', 'patient_id', 'duration'),
    )
    
    # Relationships
    patient = relationship("Patient", back_populates="sessions")
    anomalies = relationship("AnomalyLog", back_populates="session", cascade="all, delete-orphan")
