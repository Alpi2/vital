from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from .base import BaseModel


class AnomalyLog(BaseModel):
    __tablename__ = "anomaly_logs"
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("ecg_sessions.id"))
    anomaly_type = Column(String(50), nullable=False)
    severity = Column(String(20))  # low, medium, high
    confidence = Column(Float)  # 0.0 - 1.0
    timestamp = Column(DateTime, nullable=False)
    duration = Column(Float)  # seconds
    bpm_at_detection = Column(Float)
    details = Column(Text)  # JSON string with additional details
    resolved = Column(Integer, default=0)  # 0 = active, 1 = resolved
    
    # Relationships
    patient = relationship("Patient", back_populates="anomalies")
    session = relationship("ECGSession", back_populates="anomalies")
