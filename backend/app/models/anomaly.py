from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class AnomalyLog(BaseModel):
    __tablename__ = "anomaly_logs"
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("ecg_sessions.id"), index=True)
    anomaly_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), index=True)  # low, medium, high
    confidence = Column(Float)  # 0.0 - 1.0
    timestamp = Column(DateTime, nullable=False, index=True)
    duration = Column(Float)  # seconds
    bpm_at_detection = Column(Float)
    details = Column(Text)  # JSON string with additional details
    resolved = Column(Integer, default=0, index=True)  # 0 = active, 1 = resolved
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_patient_timestamp', 'patient_id', 'timestamp'),
        Index('idx_patient_type', 'patient_id', 'anomaly_type'),
        Index('idx_session_timestamp', 'session_id', 'timestamp'),
        Index('idx_type_severity', 'anomaly_type', 'severity'),
        Index('idx_timestamp_resolved', 'timestamp', 'resolved'),
    )
    
    # Relationships
    patient = relationship("Patient", back_populates="anomalies")
    session = relationship("ECGSession", back_populates="anomalies")
