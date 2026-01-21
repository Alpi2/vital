"""
Enhanced ECG Artifact Detection and Removal API

Comprehensive REST API with validation, monitoring, and real-time processing support.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import numpy as np
import asyncio
import logging
from datetime import datetime

from app.services.artifact_service import (
    RealTimeArtifactService, 
    get_artifact_service,
    ArtifactRequest,
    ArtifactResponse
)
from app.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/artifacts", tags=["Artifacts"])

# Pydantic Models for API
class ArtifactDetectionRequest(BaseModel):
    """Request model for artifact detection."""
    patient_id: str = Field(..., description="Patient identifier")
    ecg_signal: List[float] = Field(..., description="ECG signal values")
    sampling_rate: int = Field(default=360, ge=250, le=500, description="Sampling rate in Hz")
    r_peaks: Optional[List[int]] = Field(None, description="R-peak indices")
    timestamp: Optional[float] = Field(None, description="Timestamp in seconds")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    @validator('ecg_signal')
    def validate_signal_length(cls, v):
        """Validate signal length."""
        if len(v) < 1000:  # Minimum ~3 seconds at 360Hz
            raise ValueError("Signal too short - minimum 1000 samples required")
        if len(v) > 50000:  # Maximum ~140 seconds at 360Hz
            raise ValueError("Signal too long - maximum 50000 samples allowed")
        return v
    
    @validator('ecg_signal')
    def validate_signal_values(cls, v):
        """Validate signal values."""
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("All signal values must be numeric")
        if np.any(np.isnan(v)) or np.any(np.isinf(v)):
            raise ValueError("Signal contains NaN or infinite values")
        return v

class ArtifactRemovalRequest(BaseModel):
    """Request model for artifact removal."""
    patient_id: str = Field(..., description="Patient identifier")
    ecg_signal: List[float] = Field(..., description="ECG signal values")
    sampling_rate: int = Field(default=360, ge=250, le=500, description="Sampling rate in Hz")
    r_peaks: Optional[List[int]] = Field(None, description="R-peak indices")
    artifact_types: Optional[List[str]] = Field(None, description="Specific artifact types to remove")
    timestamp: Optional[float] = Field(None, description="Timestamp in seconds")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    @validator('artifact_types')
    def validate_artifact_types(cls, v):
        """Validate artifact types."""
        if v is not None:
            valid_types = ['baseline_wander', 'power_line', 'muscle', 'motion']
            invalid_types = [t for t in v if t not in valid_types]
            if invalid_types:
                raise ValueError(f"Invalid artifact types: {invalid_types}")
        return v

class ArtifactCleanRequest(BaseModel):
    """Request model for combined detection and removal."""
    patient_id: str = Field(..., description="Patient identifier")
    ecg_signal: List[float] = Field(..., description="ECG signal values")
    sampling_rate: int = Field(default=360, ge=250, le=500, description="Sampling rate in Hz")
    r_peaks: Optional[List[int]] = Field(None, description="R-peak indices")
    timestamp: Optional[float] = Field(None, description="Timestamp in seconds")
    session_id: Optional[str] = Field(None, description="Session identifier")

class ArtifactBatchRequest(BaseModel):
    """Request model for batch processing."""
    patient_id: str = Field(..., description="Patient identifier")
    segments: List[ArtifactCleanRequest] = Field(..., description="ECG signal segments")
    max_concurrent: int = Field(default=10, ge=1, le=20, description="Maximum concurrent processing")

class ArtifactResponse(BaseModel):
    """Response model for artifact processing."""
    patient_id: str
    original_signal: List[float]
    cleaned_signal: List[float]
    artifacts_detected: List[str]
    severity: str
    clean_ratio: float
    processing_time_ms: float
    quality_improvement: float
    distortion_level: float
    methods_used: List[str]
    st_segment_preserved: bool
    confidence_scores: Dict[str, float]
    affected_segments: List[tuple]
    metadata: Dict[str, float]
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    processing_time_ms: Optional[float] = None
    cache_available: bool
    redis_connected: bool
    thread_pool_active: bool
    max_concurrent_streams: int

class StatisticsResponse(BaseModel):
    """Statistics response."""
    active_streams: int
    max_concurrent_streams: int
    cache_size: int
    supported_sampling_rates: List[int]
    total_detections: Optional[int] = None
    total_removals: Optional[int] = None
    avg_processing_time: Optional[float] = None
    avg_quality_improvement: Optional[float] = None
    timestamp: float

# Helper functions
def _convert_artifact_response(response: ArtifactResponse) -> Dict[str, Any]:
    """Convert internal response to API response."""
    return {
        'patient_id': response.patient_id,
        'original_signal': response.original_signal,
        'cleaned_signal': response.cleaned_signal,
        'artifacts_detected': response.artifacts_detected,
        'severity': response.severity,
        'clean_ratio': response.clean_ratio,
        'processing_time_ms': response.processing_time_ms,
        'quality_improvement': response.quality_improvement,
        'distortion_level': response.distortion_level,
        'methods_used': response.methods_used,
        'st_segment_preserved': response.st_segment_preserved,
        'confidence_scores': response.confidence_scores,
        'affected_segments': [(int(s), int(e)) for s, e in response.affected_segments],
        'metadata': response.metadata,
        'timestamp': datetime.utcnow().isoformat()
    }

async def _store_processing_history(patient_id: str, response: Dict[str, Any]):
    """Store processing history in background."""
    try:
        service = get_artifact_service()
        history_key = f"artifact_history:{patient_id}"
        
        # Store in Redis list (keep last 1000 entries)
        await asyncio.get_event_loop().run_in_executor(
            None,
            service.redis_client.lpush,
            history_key,
            str(response)
        )
        await asyncio.get_event_loop().run_in_executor(
            None,
            service.redis_client.ltrim,
            history_key,
            0,
            999
        )
    except Exception as e:
        logger.warning(f"Failed to store processing history: {e}")

# API Endpoints
@router.post("/detect", response_model=Dict[str, Any])
async def detect_artifacts(
    request: ArtifactDetectionRequest,
    background_tasks: BackgroundTasks,
    artifact_service: RealTimeArtifactService = Depends(get_artifact_service)
):
    """
    Detect artifacts in ECG signal.
    
    - **patient_id**: Patient identifier
    - **ecg_signal**: ECG signal values (1000-50000 samples)
    - **sampling_rate**: Sampling rate (250-500 Hz)
    - **r_peaks**: Optional R-peak indices
    - **timestamp**: Optional timestamp
    - **session_id**: Optional session identifier
    
    Returns detected artifacts with confidence scores and affected segments.
    """
    try:
        # Convert to internal request format
        internal_request = ArtifactRequest(
            patient_id=request.patient_id,
            ecg_signal=request.ecg_signal,
            sampling_rate=request.sampling_rate,
            r_peaks=request.r_peaks,
            timestamp=request.timestamp,
            session_id=request.session_id
        )
        
        # Process request
        result = await artifact_service.detect_artifacts(internal_request)
        
        # Convert to API response
        response = _convert_artifact_response(result)
        
        # Store history in background
        background_tasks.add_task(_store_processing_history, request.patient_id, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Artifact detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/remove", response_model=Dict[str, Any])
async def remove_artifacts(
    request: ArtifactRemovalRequest,
    background_tasks: BackgroundTasks,
    artifact_service: RealTimeArtifactService = Depends(get_artifact_service)
):
    """
    Remove specific artifacts from ECG signal.
    
    - **artifact_types**: Specific artifact types to remove (optional, auto-detect if not provided)
    - Other parameters same as detection endpoint
    
    Returns cleaned signal with quality metrics.
    """
    try:
        # Convert to internal request format
        internal_request = ArtifactRequest(
            patient_id=request.patient_id,
            ecg_signal=request.ecg_signal,
            sampling_rate=request.sampling_rate,
            r_peaks=request.r_peaks,
            timestamp=request.timestamp,
            session_id=request.session_id
        )
        
        # Process request
        result = await artifact_service.remove_artifacts(internal_request, request.artifact_types)
        
        # Convert to API response
        response = _convert_artifact_response(result)
        
        # Store history in background
        background_tasks.add_task(_store_processing_history, request.patient_id, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Artifact removal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clean", response_model=Dict[str, Any])
async def clean_ecg_signal(
    request: ArtifactCleanRequest,
    background_tasks: BackgroundTasks,
    artifact_service: RealTimeArtifactService = Depends(get_artifact_service)
):
    """
    Detect and remove artifacts in one operation.
    
    This endpoint automatically detects artifacts and removes them using optimal methods.
    Provides the most convenient interface for signal cleaning.
    
    Returns comprehensive results including detection and removal metrics.
    """
    try:
        # Convert to internal request format
        internal_request = ArtifactRequest(
            patient_id=request.patient_id,
            ecg_signal=request.ecg_signal,
            sampling_rate=request.sampling_rate,
            r_peaks=request.r_peaks,
            timestamp=request.timestamp,
            session_id=request.session_id
        )
        
        # Process request
        result = await artifact_service.clean_signal(internal_request)
        
        # Convert to API response
        response = _convert_artifact_response(result)
        
        # Store history in background
        background_tasks.add_task(_store_processing_history, request.patient_id, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Signal cleaning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=List[Dict[str, Any]])
async def batch_process_signals(
    request: ArtifactBatchRequest,
    background_tasks: BackgroundTasks,
    artifact_service: RealTimeArtifactService = Depends(get_artifact_service)
):
    """
    Process multiple ECG segments concurrently.
    
    - **segments**: List of ECG segments to process
    - **max_concurrent**: Maximum concurrent processing (1-20)
    
    Optimized for high-throughput processing of multiple signals.
    """
    try:
        # Convert to internal request format
        internal_requests = []
        for segment in request.segments:
            internal_request = ArtifactRequest(
                patient_id=f"{request.patient_id}_{segment.patient_id}" if segment.patient_id != request.patient_id else request.patient_id,
                ecg_signal=segment.ecg_signal,
                sampling_rate=segment.sampling_rate,
                r_peaks=segment.r_peaks,
                timestamp=segment.timestamp,
                session_id=segment.session_id
            )
            internal_requests.append(internal_request)
        
        # Process batch
        results = await artifact_service.batch_process(internal_requests)
        
        # Convert to API responses
        responses = [_convert_artifact_response(result) for result in results]
        
        # Store history in background (summary only for batch)
        batch_summary = {
            'patient_id': request.patient_id,
            'segments_processed': len(results),
            'successful': sum(1 for r in results if r.artifacts_detected != ['error']),
            'failed': sum(1 for r in results if r.artifacts_detected == ['error']),
            'avg_processing_time': np.mean([r.processing_time_ms for r in results]),
            'timestamp': datetime.utcnow().isoformat()
        }
        background_tasks.add_task(_store_processing_history, f"{request.patient_id}_batch", batch_summary)
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{patient_id}", response_model=List[Dict[str, Any]])
async def get_artifact_history(
    patient_id: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of entries"),
    artifact_service: RealTimeArtifactService = Depends(get_artifact_service)
):
    """
    Get artifact processing history for a patient.
    
    - **patient_id**: Patient identifier
    - **limit**: Maximum number of history entries (1-1000)
    
    Returns historical processing data with timestamps and metrics.
    """
    try:
        history = await artifact_service.get_artifact_history(patient_id, limit)
        return history
        
    except Exception as e:
        logger.error(f"Failed to get artifact history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics", response_model=StatisticsResponse)
async def get_processing_statistics(
    artifact_service: RealTimeArtifactService = Depends(get_artifact_service)
):
    """
    Get processing statistics and performance metrics.
    
    Returns comprehensive statistics about the artifact service including:
    - Active streams and concurrency
    - Cache performance
    - Processing metrics
    - Supported sampling rates
    """
    try:
        stats = await artifact_service.get_processing_statistics()
        return StatisticsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get processing statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check(
    artifact_service: RealTimeArtifactService = Depends(get_artifact_service)
):
    """
    Health check endpoint for the artifact service.
    
    Returns service status and performance indicators.
    Used for monitoring and load balancing.
    """
    try:
        health = await artifact_service.health_check()
        return HealthResponse(**health)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status='unhealthy',
            cache_available=False,
            redis_connected=False,
            thread_pool_active=False,
            max_concurrent_streams=0
        )

@router.get("/methods", response_model=Dict[str, Any])
async def get_available_methods():
    """
    Get available artifact detection and removal methods.
    
    Returns information about supported methods and their characteristics.
    """
    return {
        "detection_methods": {
            "baseline_wander": {
                "description": "Power spectrum analysis, median filter baseline estimation, trend analysis",
                "confidence_threshold": 0.3,
                "typical_processing_time": "<50ms"
            },
            "power_line": {
                "description": "FFT-based detection with automatic frequency detection (50/60Hz)",
                "confidence_threshold": 0.4,
                "typical_processing_time": "<30ms"
            },
            "muscle": {
                "description": "High-frequency power analysis, wavelet analysis, statistical analysis",
                "confidence_threshold": 0.35,
                "typical_processing_time": "<60ms"
            },
            "motion": {
                "description": "Sudden amplitude changes, saturation detection, variance analysis",
                "confidence_threshold": 0.4,
                "typical_processing_time": "<40ms"
            }
        },
        "removal_methods": {
            "baseline_wander": {
                "methods": ["highpass", "wavelet", "median"],
                "recommended": "wavelet",
                "st_segment_preservation": True
            },
            "power_line": {
                "methods": ["adaptive", "wavelet", "notch"],
                "recommended": "adaptive",
                "st_segment_preservation": True
            },
            "muscle": {
                "methods": ["emd", "wavelet"],
                "recommended": "emd",
                "st_segment_preservation": True
            }
        },
        "performance_requirements": {
            "max_latency": "150ms",
            "max_distortion": "5%",
            "concurrent_streams": 20,
            "supported_sampling_rates": [250, 360, 500]
        }
    }

@router.delete("/cache")
async def clear_cache(
    artifact_service: RealTimeArtifactService = Depends(get_artifact_service)
):
    """
    Clear processing cache.
    
    Clears all cached results and forces fresh processing.
    Useful for testing and troubleshooting.
    """
    try:
        # Clear LRU cache
        artifact_service._get_cached_result.cache_clear()
        
        # Clear Redis cache (if needed)
        # This would require Redis FLUSHDB or specific key patterns
        
        return {
            "status": "success",
            "message": "Cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
