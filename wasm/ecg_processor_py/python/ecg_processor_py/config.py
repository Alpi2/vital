"""
ECG Analysis Configuration for Python-Rust FFI
"""

from typing import Optional
from pydantic import BaseModel, Field, validator


class ECGAnalysisConfig(BaseModel):
    """Configuration for ECG analysis with validation"""
    
    sampling_rate: float = Field(
        default=360.0,
        ge=100.0,
        le=2000.0,
        description="Sampling rate in Hz"
    )
    
    enable_hrv: bool = Field(
        default=True,
        description="Enable heart rate variability analysis"
    )
    
    enable_anomaly_detection: bool = Field(
        default=True,
        description="Enable anomaly detection"
    )
    
    min_heart_rate: float = Field(
        default=40.0,
        ge=20.0,
        le=100.0,
        description="Minimum acceptable heart rate (BPM)"
    )
    
    max_heart_rate: float = Field(
        default=200.0,
        ge=100.0,
        le=300.0,
        description="Maximum acceptable heart rate (BPM)"
    )
    
    noise_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Signal quality threshold (0-1)"
    )
    
    qrs_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="QRS detection threshold"
    )
    
    min_rr_interval: float = Field(
        default=200.0,
        ge=100.0,
        le=500.0,
        description="Minimum RR interval in ms"
    )
    
    max_rr_interval: float = Field(
        default=2000.0,
        ge=1000.0,
        le=5000.0,
        description="Maximum RR interval in ms"
    )
    
    @validator('max_heart_rate')
    def validate_heart_rate_range(cls, v, values):
        if 'min_heart_rate' in values and v <= values['min_heart_rate']:
            raise ValueError('max_heart_rate must be greater than min_heart_rate')
        return v
    
    @validator('max_rr_interval')
    def validate_rr_interval_range(cls, v, values):
        if 'min_rr_interval' in values and v <= values['min_rr_interval']:
            raise ValueError('max_rr_interval must be greater than min_rr_interval')
        return v
    
    class Config:
        extra = "forbid"
        schema_extra = {
            "example": {
                "sampling_rate": 360.0,
                "enable_hrv": True,
                "enable_anomaly_detection": True,
                "min_heart_rate": 40.0,
                "max_heart_rate": 200.0,
                "noise_threshold": 0.1,
                "qrs_threshold": 0.5,
                "min_rr_interval": 200.0,
                "max_rr_interval": 2000.0,
            }
        }
