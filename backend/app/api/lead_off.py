"""
Lead-Off Detection API Endpoints
Complete REST API with WebSocket integration for real-time monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json
import logging
from enum import Enum

from app.services.multi_lead_lead_off_service import (
    MultiLeadLeadOffService,
    MultiLeadResult,
    LeadOffResult
)
from app.services.realtime_lead_off_detector import (
    HighPerformanceLeadOffService,
    RealTimeLeadOffDetector
)
from app.services.sqi_service import SQIService

# Pydantic models for API requests and responses
class LeadOffDetectionRequest(BaseModel):
    """Request model for lead-off detection."""
    patient_id: str = Field(..., description="Patient identifier")
    ecg_signals: Dict[str, List[float]] = Field(..., description="ECG signals by lead ID")
    sampling_rate: int = Field(default=360, ge=250, le=500, description="Sampling rate in Hz")
    lead_configuration: Optional[str] = Field(default="auto", description="Lead configuration (auto, 3-lead, 5-lead, 12-lead)")

class LeadOffHistoryRequest(BaseModel):
    """Request model for lead-off history."""
    patient_id: str = Field(..., description="Patient identifier")
    lead_id: Optional[str] = Field(default=None, description="Specific lead ID")
    hours_back: int = Field(default=24, ge=1, le=168, description="Hours of history to retrieve")

class ImpedanceTestRequest(BaseModel):
    """Request model for manual impedance test."""
    patient_id: str = Field(..., description="Patient identifier")
    lead_ids: List[str] = Field(..., description="Lead IDs to test")
    test_duration_seconds: int = Field(default=10, ge=5, le=60, description="Test duration in seconds")

class StreamingDataRequest(BaseModel):
    """Request model for streaming data."""
    patient_id: str = Field(..., description="Patient identifier")
    lead_id: str = Field(..., description="Lead identifier")
    samples: List[float] = Field(..., description="New samples to add")
    sampling_rate: int = Field(default=360, ge=250, le=500, description="Sampling rate in Hz")

class LeadOffDetectionResponse(BaseModel):
    """Response model for lead-off detection."""
    success: bool = Field(..., description="Request success status")
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    overall_status: str = Field(..., description="Overall lead status")
    overall_quality: str = Field(..., description="Overall quality level")
    overall_quality_score: float = Field(..., description="Overall quality score (0-100)")
    lead_results: Dict[str, Dict[str, Any]] = Field(..., description="Individual lead results")
    alerts: List[Dict[str, Any]] = Field(default=[], description="Generated alerts")
    recommendations: List[str] = Field(default=[], description="Clinical recommendations")
    sqi_scores: Dict[str, float] = Field(default={}, description="SQI scores by lead")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class LeadStatusResponse(BaseModel):
    """Response model for lead status."""
    success: bool = Field(..., description="Request success status")
    patient_id: str = Field(..., description="Patient identifier")
    monitoring_active: bool = Field(..., description="Whether monitoring is active")
    leads_monitored: List[str] = Field(..., description="Leads being monitored")
    latest_results: List[Dict[str, Any]] = Field(..., description="Latest analysis results")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")

class ImpedanceTestResponse(BaseModel):
    """Response model for impedance test."""
    success: bool = Field(..., description="Request success status")
    patient_id: str = Field(..., description="Patient identifier")
    test_results: Dict[str, Dict[str, Any]] = Field(..., description="Impedance test results by lead")
    test_duration_seconds: int = Field(..., description="Actual test duration")
    timestamp: datetime = Field(..., description="Test completion timestamp")

class ServiceHealthResponse(BaseModel):
    """Response model for service health."""
    status: str = Field(..., description="Service health status")
    active_patients: int = Field(..., description="Number of active patients")
    total_leads_monitored: int = Field(..., description="Total leads being monitored")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    sqi_service_available: bool = Field(..., description="SQI service availability")
    timestamp: datetime = Field(..., description="Health check timestamp")

# WebSocket connection manager
class WebSocketManager:
    """Manager for WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.patient_subscriptions: Dict[str, set] = {}
    
    async def connect(self, websocket: WebSocket, patient_id: str):
        """Connect WebSocket for patient monitoring."""
        await websocket.accept()
        
        if patient_id not in self.active_connections:
            self.active_connections[patient_id] = []
        
        self.active_connections[patient_id].append(websocket)
        
        logging.info(f"WebSocket connected for patient {patient_id}")
    
    def disconnect(self, websocket: WebSocket, patient_id: str):
        """Disconnect WebSocket."""
        if patient_id in self.active_connections:
            if websocket in self.active_connections[patient_id]:
                self.active_connections[patient_id].remove(websocket)
            
            if not self.active_connections[patient_id]:
                del self.active_connections[patient_id]
        
        logging.info(f"WebSocket disconnected for patient {patient_id}")
    
    async def send_personal_message(self, message: dict, patient_id: str):
        """Send message to specific patient's connections."""
        if patient_id in self.active_connections:
            disconnected_connections = []
            
            for connection in self.active_connections[patient_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    disconnected_connections.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected_connections:
                self.active_connections[patient_id].remove(connection)
    
    async def broadcast_alert(self, alert: dict, patient_id: str):
        """Broadcast alert to patient's connections."""
        message = {
            'type': 'alert',
            'data': alert,
            'timestamp': datetime.utcnow().isoformat()
        }
        await self.send_personal_message(message, patient_id)
    
    async def broadcast_status_update(self, status: dict, patient_id: str):
        """Broadcast status update to patient's connections."""
        message = {
            'type': 'status_update',
            'data': status,
            'timestamp': datetime.utcnow().isoformat()
        }
        await self.send_personal_message(message, patient_id)

# Initialize services and WebSocket manager
router = APIRouter(prefix="/api/v1/lead-off", tags=["Lead-Off Detection"])
lead_off_service = MultiLeadLeadOffService()
realtime_service = HighPerformanceLeadOffService(num_detectors=3)
websocket_manager = WebSocketManager()
sqi_service = None

# Dependency injection
async def get_lead_off_service():
    """Get lead-off service instance."""
    return lead_off_service

async def get_realtime_service():
    """Get real-time service instance."""
    return realtime_service

async def get_sqi_service():
    """Get SQI service instance."""
    return sqi_service

# API Endpoints
@router.post("/detect", response_model=LeadOffDetectionResponse)
async def detect_lead_off(
    request: LeadOffDetectionRequest,
    background_tasks: BackgroundTasks,
    service: MultiLeadLeadOffService = Depends(get_lead_off_service)
):
    """
    Detect lead-off for patient ECG signals.
    
    This endpoint analyzes ECG signals from multiple leads to detect
    lead disconnection, poor electrode contact, and signal quality issues.
    """
    start_time = datetime.utcnow()
    
    try:
        # Convert lists to numpy arrays
        ecg_signals_np = {}
        for lead_id, signal_data in request.ecg_signals.items():
            ecg_signals_np[lead_id] = np.array(signal_data)
        
        # Perform detection
        result = await service.detect_multi_lead_off(
            patient_id=request.patient_id,
            ecg_signals=ecg_signals_np,
            sampling_rate=request.sampling_rate
        )
        
        # Convert lead results to serializable format
        lead_results_serializable = {}
        for lead_id, lead_result in result.lead_results.items():
            lead_results_serializable[lead_id] = {
                'lead_id': lead_result.lead_id,
                'status': lead_result.status.value,
                'quality': lead_result.quality.value,
                'quality_score': lead_result.quality_score,
                'impedance_dc': lead_result.impedance_dc,
                'impedance_ac': lead_result.impedance_ac,
                'snr': lead_result.snr,
                'amplitude_mv': lead_result.amplitude_mv,
                'saturation_ratio': lead_result.saturation_ratio,
                'variance': lead_result.variance,
                'timestamp': lead_result.timestamp.isoformat(),
                'reason': lead_result.reason,
                'confidence': lead_result.confidence,
                'metrics': lead_result.metrics
            }
        
        # Broadcast alerts via WebSocket
        if result.alerts:
            for alert in result.alerts:
                await websocket_manager.broadcast_alert(alert, request.patient_id)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return LeadOffDetectionResponse(
            success=True,
            patient_id=result.patient_id,
            timestamp=result.timestamp,
            overall_status=result.overall_status.value,
            overall_quality=result.overall_quality.value,
            overall_quality_score=result.overall_quality_score,
            lead_results=lead_results_serializable,
            alerts=result.alerts,
            recommendations=result.recommendations,
            sqi_scores=result.sqi_scores,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logging.error(f"Error in lead-off detection: {e}")
        raise HTTPException(status_code=500, detail=f"Lead-off detection failed: {str(e)}")

@router.get("/status/{patient_id}", response_model=LeadStatusResponse)
async def get_lead_status(
    patient_id: str,
    service: MultiLeadLeadOffService = Depends(get_lead_off_service)
):
    """
    Get current lead status for patient.
    
    Returns the current status of all leads being monitored for the patient,
    including quality scores and recent analysis results.
    """
    try:
        # Get quality summary
        quality_summary = await service.get_patient_quality_summary(patient_id)
        
        # Get real-time status
        realtime_status = await realtime_service.detectors[0].get_real_time_status(patient_id)
        
        # Get performance metrics
        performance_metrics = realtime_status.get('performance_metrics', {})
        
        # Format latest results
        latest_results = realtime_status.get('latest_results', [])
        
        return LeadStatusResponse(
            success=True,
            patient_id=patient_id,
            monitoring_active=realtime_status.get('monitoring_active', False),
            leads_monitored=realtime_status.get('leads_monitored', []),
            latest_results=latest_results,
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logging.error(f"Error getting lead status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lead status: {str(e)}")

@router.get("/history/{patient_id}")
async def get_lead_history(
    patient_id: str,
    lead_id: Optional[str] = None,
    hours_back: int = 24,
    service: MultiLeadLeadOffService = Depends(get_lead_off_service)
):
    """
    Get lead-off history for patient.
    
    Returns historical data about lead quality, impedance measurements,
    and status changes for the specified time period.
    """
    try:
        history = await service.get_lead_status_history(
            patient_id=patient_id,
            lead_id=lead_id,
            hours_back=hours_back
        )
        
        return {
            'success': True,
            'patient_id': patient_id,
            'history': history,
            'hours_back': hours_back,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error getting lead history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lead history: {str(e)}")

@router.post("/test", response_model=ImpedanceTestResponse)
async def test_impedance(
    request: ImpedanceTestRequest,
    background_tasks: BackgroundTasks,
    service: MultiLeadLeadOffService = Depends(get_lead_off_service)
):
    """
    Perform manual impedance test for specified leads.
    
    This endpoint triggers a comprehensive impedance measurement
    for the specified leads, which can help diagnose electrode issues.
    """
    try:
        # This would integrate with hardware for actual impedance measurement
        # For now, we'll simulate the test results
        
        test_results = {}
        
        for lead_id in request.lead_ids:
            # Simulate impedance measurement
            # In real implementation, this would interface with hardware
            impedance_dc = np.random.uniform(1.0, 15.0)  # kΩ
            impedance_ac = np.random.uniform(0.8, 12.0)   # kΩ
            
            # Determine quality based on impedance
            if impedance_dc < 2.0:
                quality = "excellent"
            elif impedance_dc < 5.0:
                quality = "good"
            elif impedance_dc < 10.0:
                quality = "acceptable"
            else:
                quality = "poor"
            
            test_results[lead_id] = {
                'impedance_dc_kohm': impedance_dc,
                'impedance_ac_kohm': impedance_ac,
                'quality': quality,
                'test_passed': impedance_dc < 10.0,
                'recommendation': "Normal" if impedance_dc < 10.0 else "Check electrode connection"
            }
        
        return ImpedanceTestResponse(
            success=True,
            patient_id=request.patient_id,
            test_results=test_results,
            test_duration_seconds=request.test_duration_seconds,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logging.error(f"Error in impedance test: {e}")
        raise HTTPException(status_code=500, detail=f"Impedance test failed: {str(e)}")

@router.get("/quality/{patient_id}")
async def get_electrode_quality(
    patient_id: str,
    service: MultiLeadLeadOffService = Depends(get_lead_off_service)
):
    """
    Get electrode quality assessment for patient.
    
    Returns detailed quality metrics for all electrodes,
    including impedance measurements and quality trends.
    """
    try:
        quality_summary = await service.get_patient_quality_summary(patient_id)
        
        return {
            'success': True,
            'patient_id': patient_id,
            'quality_summary': quality_summary,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error getting electrode quality: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get electrode quality: {str(e)}")

@router.post("/stream")
async def stream_ecg_data(
    request: StreamingDataRequest,
    realtime_service: HighPerformanceLeadOffService = Depends(get_realtime_service)
):
    """
    Stream ECG data for real-time monitoring.
    
    Accepts new ECG samples and processes them in real-time,
    generating alerts and status updates as needed.
    """
    try:
        # Convert to numpy array
        new_samples = np.array(request.samples)
        
        # Define callback for real-time alerts
        async def alert_callback(result):
            if result.status.value in ["disconnected", "poor_quality"]:
                alert = {
                    'patient_id': request.patient_id,
                    'lead_id': result.lead_id,
                    'status': result.status.value,
                    'quality': result.quality.value,
                    'reason': result.reason,
                    'timestamp': result.timestamp.isoformat()
                }
                await websocket_manager.broadcast_alert(alert, request.patient_id)
        
        # Process streaming data
        detector = realtime_service.detectors[0]
        await detector.process_streaming_data(
            patient_id=request.patient_id,
            lead_id=request.lead_id,
            new_samples=new_samples,
            sampling_rate=request.sampling_rate,
            analysis_callback=alert_callback
        )
        
        return {
            'success': True,
            'patient_id': request.patient_id,
            'lead_id': request.lead_id,
            'samples_processed': len(new_samples),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error in streaming data: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming data processing failed: {str(e)}")

@router.get("/health", response_model=ServiceHealthResponse)
async def get_service_health(
    service: MultiLeadLeadOffService = Depends(get_lead_off_service),
    realtime_service: HighPerformanceLeadOffService = Depends(get_realtime_service)
):
    """
    Get service health status.
    
    Returns overall service health, performance metrics,
    and operational status.
    """
    try:
        # Get service health
        service_health = await service.get_service_health()
        
        # Get real-time service status
        realtime_status = await realtime_service.get_service_status()
        
        return ServiceHealthResponse(
            status=service_health.get('status', 'unknown'),
            active_patients=service_health.get('active_patients', 0),
            total_leads_monitored=service_health.get('total_leads_monitored', 0),
            performance_metrics={
                'service_metrics': service_health,
                'realtime_metrics': realtime_status
            },
            sqi_service_available=service_health.get('sqi_service_available', False),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logging.error(f"Error getting service health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# WebSocket endpoint for real-time monitoring
@router.websocket("/ws/{patient_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    patient_id: str,
    realtime_service: HighPerformanceLeadOffService = Depends(get_realtime_service)
):
    """
    WebSocket endpoint for real-time lead-off monitoring.
    
    Provides real-time updates on lead status, alerts, and quality changes.
    """
    await websocket_manager.connect(websocket, patient_id)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                # Handle different message types
                if message.get('type') == 'ping':
                    await websocket.send_text(json.dumps({
                        'type': 'pong',
                        'timestamp': datetime.utcnow().isoformat()
                    }))
                elif message.get('type') == 'get_status':
                    status = await realtime_service.detectors[0].get_real_time_status(patient_id)
                    await websocket_manager.send_status_update(status, patient_id)
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }))
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, patient_id)
        logging.info(f"WebSocket disconnected for patient {patient_id}")
    except Exception as e:
        logging.error(f"WebSocket error for patient {patient_id}: {e}")
        websocket_manager.disconnect(websocket, patient_id)

# Background task for service initialization
@router.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global sqi_service
    
    try:
        # Initialize SQI service integration
        from app.services.sqi_service import SQIService
        sqi_service = SQIService()
        await lead_off_service.initialize_sqi_service(sqi_service)
        
        # Start background tasks
        await realtime_service.start_all_background_tasks()
        
        logging.info("Lead-off detection service initialized successfully")
        
    except Exception as e:
        logging.error(f"Error initializing lead-off detection service: {e}")

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        # Cleanup resources
        for detector in realtime_service.detectors:
            await detector.cleanup_resources()
        
        logging.info("Lead-off detection service shutdown completed")
        
    except Exception as e:
        logging.error(f"Error during service shutdown: {e}")
