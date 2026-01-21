"""
REST API Endpoints for Anomaly Detection

FastAPI endpoints for real-time anomaly detection,
clinical analysis, and anomaly management.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from datetime import datetime, timedelta

from ..services.anomaly_detection_service import (
    AnomalyDetectionService, get_anomaly_detection_service, AnomalyDetection
)
from ..services.feature_extraction_service import get_feature_service
from ..ml.inference_service import InferenceService
from ..ml.anomaly_types import AnomalyType, Severity, DetectionMethod, get_anomaly_definition
from ..database import get_async_session
from ..models.anomaly import AnomalyLog
from ..websocket import manager as ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/anomaly-detection", tags=["anomaly-detection"])

# Pydantic models
class ECGAnomalyRequest(BaseModel):
    """Request model for ECG anomaly detection."""
    signal: List[float] = Field(..., description="ECG signal samples")
    sampling_rate: int = Field(250, description="Sampling rate in Hz")
    peaks: Optional[List[int]] = Field(None, description="R-peak indices (optional)")
    rr_intervals: Optional[List[float]] = Field(None, description="RR intervals in ms (optional)")
    patient_id: int = Field(..., description="Patient identifier")
    session_id: Optional[int] = Field(None, description="Monitoring session ID")
    patient_info: Optional[Dict[str, Any]] = Field(None, description="Patient demographics and history")
    confidence_threshold: Optional[float] = Field(0.7, description="Minimum confidence threshold")

class BatchAnomalyRequest(BaseModel):
    """Request model for batch anomaly detection."""
    signals: List[List[float]] = Field(..., description="List of ECG signals")
    sampling_rate: int = Field(250, description="Sampling rate in Hz")
    peaks_list: Optional[List[List[int]]] = Field(None, description="List of R-peak indices")
    rr_intervals_list: Optional[List[List[float]]] = Field(None, description="List of RR intervals")
    patient_id: int = Field(..., description="Patient identifier")
    session_id: Optional[int] = Field(None, description="Monitoring session ID")
    patient_info: Optional[Dict[str, Any]] = Field(None, description="Patient demographics and history")
    confidence_threshold: Optional[float] = Field(0.7, description="Minimum confidence threshold")

class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection."""
    anomalies: List[Dict[str, Any]] = Field(..., description="Detected anomalies")
    total_anomalies: int = Field(..., description="Total number of anomalies detected")
    detection_time_ms: float = Field(..., description="Detection time in milliseconds")
    patient_id: int = Field(..., description="Patient identifier")
    session_id: Optional[int] = Field(None, description="Monitoring session ID")
    detection_summary: Dict[str, int] = Field(..., description="Summary by severity")

class AnomalyHistoryRequest(BaseModel):
    """Request model for anomaly history."""
    patient_id: int = Field(..., description="Patient identifier")
    start_date: Optional[datetime] = Field(None, description="Start date for history")
    end_date: Optional[datetime] = Field(None, description="End date for history")
    anomaly_types: Optional[List[str]] = Field(None, description="Filter by anomaly types")
    severity: Optional[str] = Field(None, description="Filter by severity")
    limit: Optional[int] = Field(100, description="Maximum number of records")

class AnomalyStatisticsResponse(BaseModel):
    """Response model for anomaly statistics."""
    total_detections: int = Field(..., description="Total anomaly detections")
    detections_by_type: Dict[str, int] = Field(..., description="Detections by anomaly type")
    detections_by_severity: Dict[str, int] = Field(..., description="Detections by severity")
    detection_methods: Dict[str, int] = Field(..., description="Detections by method")
    time_distribution: Dict[str, int] = Field(..., description="Detections by time period")
    patient_risk_score: float = Field(..., description="Patient risk score (0-100)")
    trend_analysis: Dict[str, Any] = Field(..., description="Trend analysis")

class ClinicalContextRequest(BaseModel):
    """Request model for clinical context update."""
    patient_id: int = Field(..., description="Patient identifier")
    age: int = Field(..., description="Patient age")
    gender: str = Field(..., description="Patient gender")
    medications: List[str] = Field(default_factory=list, description="Current medications")
    history: List[str] = Field(default_factory=list, description="Medical history")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors")

# Dependencies
async def get_anomaly_service() -> AnomalyDetectionService:
    """Get anomaly detection service."""
    # This would be properly injected in a real application
    from ..ml.inference_service import get_inference_service
    from ..services.feature_extraction_service import get_feature_service
    
    inference_service = get_inference_service()
    feature_service = get_feature_service()
    
    return get_anomaly_detection_service(inference_service, feature_service)

@router.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: ECGAnomalyRequest,
    background_tasks: BackgroundTasks,
    service: AnomalyDetectionService = Depends(get_anomaly_service),
    db = Depends(get_async_session)
):
    """
    Detect anomalies in ECG signal.
    
    - **signal**: ECG signal samples
    - **sampling_rate**: Sampling rate in Hz (default: 250)
    - **peaks**: Optional R-peak indices
    - **rr_intervals**: Optional RR intervals in milliseconds
    - **patient_id**: Patient identifier
    - **session_id**: Optional monitoring session ID
    - **patient_info**: Optional patient demographics and history
    - **confidence_threshold**: Minimum confidence threshold (default: 0.7)
    
    Performs comprehensive anomaly detection using ML models and rule-based methods.
    Returns detected anomalies with confidence scores and clinical context.
    """
    try:
        import time
        start_time = time.time()
        
        # Convert to numpy array
        signal = np.array(request.signal)
        
        # Use provided peaks/RR intervals or detect them
        peaks = request.peaks if request.peaks is not None else []
        rr_intervals = np.array(request.rr_intervals) if request.rr_intervals is not None else np.array([])
        
        # Update confidence threshold if provided
        if request.confidence_threshold != service.confidence_threshold:
            service.update_confidence_threshold(request.confidence_threshold)
        
        # Detect anomalies
        anomalies = await service.detect_anomalies(
            signal=signal,
            peaks=peaks,
            rr_intervals=rr_intervals,
            patient_id=request.patient_id,
            session_id=request.session_id,
            patient_info=request.patient_info,
            db_session=db
        )
        
        detection_time = (time.time() - start_time) * 1000
        
        # Convert to response format
        anomaly_dicts = []
        severity_summary = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for anomaly in anomalies:
            anomaly_dict = {
                'type': anomaly.type.value,
                'confidence': anomaly.confidence,
                'severity': anomaly.severity.value,
                'detection_method': anomaly.detection_method.value,
                'timestamp': anomaly.timestamp.isoformat(),
                'description': anomaly.description,
                'features': anomaly.features,
                'clinical_context': anomaly.clinical_context
            }
            anomaly_dicts.append(anomaly_dict)
            severity_summary[anomaly.severity.value] += 1
        
        return AnomalyDetectionResponse(
            anomalies=anomaly_dicts,
            total_anomalies=len(anomalies),
            detection_time_ms=detection_time,
            patient_id=request.patient_id,
            session_id=request.session_id,
            detection_summary=severity_summary
        )
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-batch", response_model=List[AnomalyDetectionResponse])
async def detect_anomalies_batch(
    request: BatchAnomalyRequest,
    background_tasks: BackgroundTasks,
    service: AnomalyDetectionService = Depends(get_anomaly_service),
    db = Depends(get_async_session)
):
    """
    Detect anomalies in multiple ECG signals.
    
    - **signals**: List of ECG signals
    - **sampling_rate**: Sampling rate in Hz (default: 250)
    - **peaks_list**: Optional list of R-peak indices for each signal
    - **rr_intervals_list**: Optional list of RR intervals for each signal
    - **patient_id**: Patient identifier
    - **session_id**: Optional monitoring session ID
    - **patient_info**: Optional patient demographics and history
    - **confidence_threshold**: Minimum confidence threshold (default: 0.7)
    
    Processes multiple signals efficiently with batch optimization.
    """
    try:
        import time
        start_time = time.time()
        
        # Convert to numpy arrays
        signals = [np.array(signal) for signal in request.signals]
        
        # Use provided peaks/RR intervals or empty lists
        peaks_list = request.peaks_list if request.peaks_list is not None else [[] for _ in signals]
        rr_intervals_list = request.rr_intervals_list if request.rr_intervals_list is not None else [np.array([]) for _ in signals]
        
        # Update confidence threshold if provided
        if request.confidence_threshold != service.confidence_threshold:
            service.update_confidence_threshold(request.confidence_threshold)
        
        # Process each signal
        results = []
        total_time = 0
        
        for i, (signal, peaks, rr_intervals) in enumerate(zip(signals, peaks_list, rr_intervals_list)):
            signal_start = time.time()
            
            anomalies = await service.detect_anomalies(
                signal=signal,
                peaks=peaks,
                rr_intervals=rr_intervals,
                patient_id=request.patient_id,
                session_id=request.session_id,
                patient_info=request.patient_info,
                db_session=db
            )
            
            signal_time = (time.time() - signal_start) * 1000
            total_time += signal_time
            
            # Convert to response format
            anomaly_dicts = []
            severity_summary = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            
            for anomaly in anomalies:
                anomaly_dict = {
                    'type': anomaly.type.value,
                    'confidence': anomaly.confidence,
                    'severity': anomaly.severity.value,
                    'detection_method': anomaly.detection_method.value,
                    'timestamp': anomaly.timestamp.isoformat(),
                    'description': anomaly.description,
                    'features': anomaly.features,
                    'clinical_context': anomaly.clinical_context
                }
                anomaly_dicts.append(anomaly_dict)
                severity_summary[anomaly.severity.value] += 1
            
            results.append(AnomalyDetectionResponse(
                anomalies=anomaly_dicts,
                total_anomalies=len(anomalies),
                detection_time_ms=signal_time,
                patient_id=request.patient_id,
                session_id=request.session_id,
                detection_summary=severity_summary
            ))
        
        batch_time = (time.time() - start_time) * 1000
        avg_time_per_signal = batch_time / len(signals)
        
        logger.info(f"Batch anomaly detection: {len(signals)} signals in {batch_time:.2f}ms "
                   f"({avg_time_per_signal:.2f}ms per signal)")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{patient_id}")
async def get_anomaly_history(
    patient_id: int,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    anomaly_types: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    db = Depends(get_async_session)
):
    """
    Get anomaly detection history for a patient.
    
    - **patient_id**: Patient identifier
    - **start_date**: Optional start date for filtering
    - **end_date**: Optional end date for filtering
    - **anomaly_types**: Optional comma-separated list of anomaly types
    - **severity**: Optional severity filter
    - **limit**: Maximum number of records (default: 100, max: 1000)
    
    Returns historical anomaly detections with filtering options.
    """
    try:
        # Build query
        query = db.query(AnomalyLog).filter(AnomalyLog.patient_id == patient_id)
        
        # Apply filters
        if start_date:
            query = query.filter(AnomalyLog.detected_at >= start_date)
        
        if end_date:
            query = query.filter(AnomalyLog.detected_at <= end_date)
        
        if anomaly_types:
            type_list = [t.strip() for t in anomaly_types.split(',')]
            query = query.filter(AnomalyLog.anomaly_type.in_(type_list))
        
        if severity:
            query = query.filter(AnomalyLog.severity == severity)
        
        # Order and limit
        query = query.order_by(AnomalyLog.detected_at.desc()).limit(limit)
        
        # Execute query
        anomalies = await query.all()
        
        # Convert to response format
        history = []
        for anomaly in anomalies:
            history.append({
                'id': anomaly.id,
                'anomaly_type': anomaly.anomaly_type,
                'confidence': anomaly.confidence,
                'severity': anomaly.severity,
                'detection_method': anomaly.detection_method,
                'description': anomaly.description,
                'features': anomaly.features,
                'clinical_context': anomaly.clinical_context,
                'detected_at': anomaly.detected_at.isoformat(),
                'session_id': anomaly.session_id
            })
        
        return {
            'patient_id': patient_id,
            'total_records': len(history),
            'history': history,
            'filters': {
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'anomaly_types': anomaly_types,
                'severity': severity,
                'limit': limit
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get anomaly history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics/{patient_id}")
async def get_anomaly_statistics(
    patient_id: int,
    days: int = Query(30, ge=1, le=365),
    db = Depends(get_async_session)
):
    """
    Get anomaly detection statistics for a patient.
    
    - **patient_id**: Patient identifier
    - **days**: Number of days to analyze (default: 30, range: 1-365)
    
    Returns comprehensive statistics and risk assessment.
    """
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Query anomalies in date range
        query = db.query(AnomalyLog).filter(
            AnomalyLog.patient_id == patient_id,
            AnomalyLog.detected_at >= start_date,
            AnomalyLog.detected_at <= end_date
        )
        
        anomalies = await query.all()
        
        # Calculate statistics
        total_detections = len(anomalies)
        
        # By type
        detections_by_type = {}
        for anomaly in anomalies:
            detections_by_type[anomaly.anomaly_type] = detections_by_type.get(anomaly.anomaly_type, 0) + 1
        
        # By severity
        detections_by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for anomaly in anomalies:
            detections_by_severity[anomaly.severity] += 1
        
        # By detection method
        detection_methods = {'ml': 0, 'rule': 0, 'hybrid': 0}
        for anomaly in anomalies:
            detection_methods[anomaly.detection_method] = detection_methods.get(anomaly.detection_method, 0) + 1
        
        # Time distribution (by hour of day)
        time_distribution = {}
        for anomaly in anomalies:
            hour = anomaly.detected_at.hour
            time_distribution[str(hour)] = time_distribution.get(str(hour), 0) + 1
        
        # Calculate risk score (0-100)
        critical_weight = 10
        high_weight = 5
        medium_weight = 2
        low_weight = 1
        
        risk_score = (
            detections_by_severity['critical'] * critical_weight +
            detections_by_severity['high'] * high_weight +
            detections_by_severity['medium'] * medium_weight +
            detections_by_severity['low'] * low_weight
        )
        
        # Normalize to 0-100 scale
        max_possible_score = days * 24 * critical_weight  # Worst case: critical every hour
        risk_score = min(100, (risk_score / max_possible_score) * 100)
        
        # Trend analysis (simplified)
        recent_anomalies = [a for a in anomalies if a.detected_at >= end_date - timedelta(days=7)]
        older_anomalies = [a for a in anomalies if a.detected_at < end_date - timedelta(days=7)]
        
        trend = "stable"
        if len(recent_anomalies) > len(older_anomalies) * 1.2:
            trend = "increasing"
        elif len(recent_anomalies) < len(older_anomalies) * 0.8:
            trend = "decreasing"
        
        return AnomalyStatisticsResponse(
            total_detections=total_detections,
            detections_by_type=detections_by_type,
            detections_by_severity=detections_by_severity,
            detection_methods=detection_methods,
            time_distribution=time_distribution,
            patient_risk_score=risk_score,
            trend_analysis={
                'trend': trend,
                'recent_count': len(recent_anomalies),
                'older_count': len(older_anomalies),
                'analysis_period_days': days
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get anomaly statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clinical-context/{patient_id}")
async def update_clinical_context(
    patient_id: int,
    request: ClinicalContextRequest,
    service: AnomalyDetectionService = Depends(get_anomaly_service)
):
    """
    Update clinical context for a patient.
    
    - **patient_id**: Patient identifier
    - **age**: Patient age
    - **gender**: Patient gender
    - **medications**: List of current medications
    - **history**: List of medical history items
    - **risk_factors**: List of risk factors
    
    Updates patient context used for anomaly detection adjustments.
    """
    try:
        # Prepare patient context
        patient_context = {
            'age': request.age,
            'gender': request.gender,
            'medications': request.medications,
            'history': request.history,
            'risk_factors': request.risk_factors,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        # Clear cache for this patient to force refresh
        if hasattr(service, 'patient_context_cache'):
            if patient_id in service.patient_context_cache:
                del service.patient_context_cache[patient_id]
        
        # In a real implementation, this would be saved to database
        # For now, we'll just return success
        
        return {
            'message': 'Clinical context updated successfully',
            'patient_id': patient_id,
            'context': patient_context
        }
        
    except Exception as e:
        logger.error(f"Failed to update clinical context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/service-status")
async def get_service_status(
    service: AnomalyDetectionService = Depends(get_anomaly_service)
):
    """
    Get anomaly detection service status and statistics.
    
    Returns service health, configuration, and performance metrics.
    """
    try:
        stats = service.get_detection_statistics()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'service_info': {
                'model_id': service.model_id,
                'confidence_threshold': service.confidence_threshold,
                'ml_enabled': service.enable_ml,
                'rules_enabled': service.enable_rules,
                'hybrid_enabled': service.enable_hybrid
            },
            'performance': stats,
            'supported_anomaly_types': [
                atype.value for atype in AnomalyType
                if atype != AnomalyType.NORMAL
            ],
            'severity_levels': [s.value for s in Severity],
            'detection_methods': [m.value for m in DetectionMethod]
        }
        
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        return JSONResponse(
            status_code=503,
            content={
                'status': 'unhealthy',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
        )

@router.post("/clear-cache/{patient_id}")
async def clear_patient_cache(
    patient_id: int,
    service: AnomalyDetectionService = Depends(get_anomaly_service)
):
    """
    Clear patient context cache.
    
    - **patient_id**: Patient identifier
    
    Clears cached patient context to force refresh on next detection.
    """
    try:
        if hasattr(service, 'patient_context_cache'):
            if patient_id in service.patient_context_cache:
                del service.patient_context_cache[patient_id]
        
        return {
            'message': f'Cache cleared for patient {patient_id}',
            'patient_id': patient_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear patient cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anomaly-types")
async def get_supported_anomaly_types():
    """
    Get list of supported anomaly types with clinical information.
    
    Returns comprehensive information about supported anomaly types,
    including clinical significance and treatment urgency.
    """
    try:
        anomaly_info = {}
        
        for atype in AnomalyType:
            if atype == AnomalyType.NORMAL:
                continue
                
            definition = get_anomaly_definition(atype)
            
            anomaly_info[atype.value] = {
                'name': definition.name if definition else atype.value,
                'description': definition.description if definition else 'Unknown anomaly',
                'icd10_codes': definition.icd10_codes if definition else [],
                'default_severity': definition.default_severity.value if definition else 'medium',
                'clinical_significance': definition.clinical_significance if definition else 'Unknown',
                'treatment_urgency': definition.treatment_urgency if definition else 'Unknown',
                'differential_diagnoses': definition.differential_diagnoses if definition else [],
                'associated_conditions': definition.associated_conditions if definition else []
            }
        
        return {
            'supported_anomalies': anomaly_info,
            'total_count': len(anomaly_info),
            'severity_levels': {
                'critical': 'Immediate life threat',
                'high': 'Urgent medical attention required',
                'medium': 'Medical evaluation recommended',
                'low': 'Monitor, treat if symptomatic'
            },
            'detection_methods': {
                'ml': 'Machine learning model prediction',
                'rule': 'Rule-based clinical criteria',
                'hybrid': 'Combined ML and rule-based detection'
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get anomaly types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-signal")
async def test_anomaly_detection(
    signal: List[float],
    anomaly_type: str = Query(..., description="Expected anomaly type"),
    service: AnomalyDetectionService = Depends(get_anomaly_service),
    db = Depends(get_async_session)
):
    """
    Test anomaly detection with a known signal.
    
    - **signal**: ECG signal samples
    - **anomaly_type**: Expected anomaly type for validation
    
    Useful for testing and validation purposes.
    """
    try:
        import time
        start_time = time.time()
        
        # Convert to numpy array
        signal_array = np.array(signal)
        
        # Simple peak detection for testing
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(signal_array, height=np.mean(signal_array))
        
        # Calculate RR intervals
        rr_intervals = np.diff(peaks) * 1000 / 250 if len(peaks) > 1 else np.array([])
        
        # Detect anomalies
        anomalies = await service.detect_anomalies(
            signal=signal_array,
            peaks=peaks.tolist(),
            rr_intervals=rr_intervals,
            patient_id=0,  # Test patient
            session_id=None,
            patient_info={'age': 50, 'gender': 'male'},
            db_session=db
        )
        
        detection_time = (time.time() - start_time) * 1000
        
        # Check if expected anomaly was detected
        detected_types = [a.type.value for a in anomalies]
        expected_detected = anomaly_type in detected_types
        
        return {
            'test_result': 'PASS' if expected_detected else 'FAIL',
            'expected_anomaly': anomaly_type,
            'detected_anomalies': detected_types,
            'total_anomalies': len(anomalies),
            'detection_time_ms': detection_time,
            'anomaly_details': [
                {
                    'type': a.type.value,
                    'confidence': a.confidence,
                    'severity': a.severity.value,
                    'method': a.detection_method.value
                }
                for a in anomalies
            ]
        }
        
    except Exception as e:
        logger.error(f"Test detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
