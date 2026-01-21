"""
FastAPI integration for gRPC Alarm Client
Provides REST API endpoints for alarm management using gRPC backend
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
import asyncio
import json
import logging
from datetime import datetime

# Import gRPC client and exceptions
from app.services.grpc.alarm_client import AlarmGRPCClient
from app.exceptions.grpc_exceptions import (
    AlarmServiceError, CircuitBreakerOpenError, GRPCTimeoutError,
    GRPCUnavailableError, InvalidAlarmTypeError, InvalidAlarmSeverityError
)

# Import generated protobufs
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../generated/protos'))

try:
    from alarm.v1 import alarm_pb2
    from common.v1 import common_pb2
except ImportError as e:
    raise ImportError(f"Generated protobufs not available: {e}")

# Import security dependencies
from app.security.dependencies import get_current_user
from app.models.user import User

# Import Pydantic models
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/alarms", tags=["alarms"])

# Global client instance (will be managed by lifespan events)
alarm_client: Optional[AlarmGRPCClient] = None


# Pydantic Models
class AlarmMetadata(BaseModel):
    """Alarm metadata model"""
    device_id: Optional[str] = None
    heart_rate: Optional[str] = None
    threshold: Optional[str] = None
    location: Optional[str] = None
    custom_data: Optional[Dict[str, Any]] = None


class CreateAlarmRequest(BaseModel):
    """Create alarm request model"""
    patient_id: str = Field(..., description="Patient ID")
    alarm_type: str = Field(..., description="Alarm type (e.g., 'tachycardia', 'bradycardia')")
    severity: str = Field(..., description="Alarm severity (e.g., 'low', 'medium', 'high', 'critical')")
    message: str = Field(..., description="Alarm message")
    description: Optional[str] = Field(None, description="Detailed alarm description")
    source: Optional[str] = Field(None, description="Alarm source")
    device_id: Optional[str] = Field(None, description="Device ID that triggered the alarm")
    metadata: Optional[AlarmMetadata] = Field(None, description="Additional alarm metadata")
    
    @validator('alarm_type')
    def validate_alarm_type(cls, v):
        """Validate alarm type"""
        valid_types = [
            'tachycardia', 'bradycardia', 'afib', 'vfib', 'pvc',
            'st_elevation', 'st_depression', 'lead_off', 'signal_quality',
            'device_error', 'battery_low', 'disconnected', 'threshold_breach'
        ]
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid alarm type. Must be one of: {', '.join(valid_types)}")
        return v.lower()
    
    @validator('severity')
    def validate_severity(cls, v):
        """Validate alarm severity"""
        valid_severities = ['info', 'low', 'medium', 'high', 'critical']
        if v.lower() not in valid_severities:
            raise ValueError(f"Invalid severity. Must be one of: {', '.join(valid_severities)}")
        return v.lower()


class AcknowledgeAlarmRequest(BaseModel):
    """Acknowledge alarm request model"""
    alarm_id: str = Field(..., description="Alarm ID to acknowledge")
    notes: Optional[str] = Field(None, description="Acknowledgment notes")


class ResolveAlarmRequest(BaseModel):
    """Resolve alarm request model"""
    alarm_id: str = Field(..., description="Alarm ID to resolve")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")


class AlarmResponse(BaseModel):
    """Alarm response model"""
    id: str
    patient_id: str
    type: str
    severity: str
    status: str
    message: str
    description: Optional[str]
    source: Optional[str]
    device_id: Optional[str]
    created_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    acknowledged_by: Optional[str]
    resolved_by: Optional[str]
    escalation_level: int
    metadata: Dict[str, str]
    
    class Config:
        from_attributes = True


class GetAlarmsQuery(BaseModel):
    """Query parameters for getting alarms"""
    patient_id: Optional[str] = None
    status: Optional[str] = None
    severity: Optional[str] = None
    alarm_type: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


# Helper functions
def protobuf_to_datetime(timestamp) -> datetime:
    """Convert protobuf timestamp to datetime"""
    return datetime.fromtimestamp(timestamp.seconds + timestamp.nanos / 1e9)


def alarm_to_response(alarm: alarm_pb2.Alarm) -> AlarmResponse:
    """Convert protobuf alarm to response model"""
    return AlarmResponse(
        id=alarm.id,
        patient_id=alarm.patient_id,
        type=alarm_pb2.AlarmType.Name(alarm.type).replace('ALARM_TYPE_', '').lower(),
        severity=alarm_pb2.AlarmSeverity.Name(alarm.severity).replace('ALARM_SEVERITY_', '').lower(),
        status=alarm_pb2.AlarmStatus.Name(alarm.status).replace('ALARM_STATUS_', '').lower(),
        message=alarm.message,
        description=alarm.description if alarm.description else None,
        source=alarm.source if alarm.source else None,
        device_id=alarm.device_id if alarm.device_id else None,
        created_at=protobuf_to_datetime(alarm.created_at),
        acknowledged_at=protobuf_to_datetime(alarm.acknowledged_at) if alarm.HasField('acknowledged_at') else None,
        resolved_at=protobuf_to_datetime(alarm.resolved_at) if alarm.HasField('resolved_at') else None,
        acknowledged_by=alarm.acknowledged_by if alarm.acknowledged_by else None,
        resolved_by=alarm.resolved_by if alarm.resolved_by else None,
        escalation_level=alarm.escalation_level,
        metadata=dict(alarm.metadata)
    )


def metadata_to_dict(metadata: Optional[AlarmMetadata]) -> Dict[str, str]:
    """Convert metadata model to dict"""
    if not metadata:
        return {}
    
    result = {}
    if metadata.device_id:
        result['device_id'] = metadata.device_id
    if metadata.heart_rate:
        result['heart_rate'] = metadata.heart_rate
    if metadata.threshold:
        result['threshold'] = metadata.threshold
    if metadata.location:
        result['location'] = metadata.location
    if metadata.custom_data:
        result.update({k: str(v) for k, v in metadata.custom_data.items()})
    
    return result


# Error handling
def handle_grpc_error(error: Exception) -> HTTPException:
    """Convert gRPC errors to HTTP exceptions"""
    if isinstance(error, InvalidAlarmTypeError):
        return HTTPException(status_code=400, detail=str(error))
    elif isinstance(error, InvalidAlarmSeverityError):
        return HTTPException(status_code=400, detail=str(error))
    elif isinstance(error, CircuitBreakerOpenError):
        return HTTPException(status_code=503, detail="Alarm service temporarily unavailable")
    elif isinstance(error, GRPCTimeoutError):
        return HTTPException(status_code=408, detail="Request timeout")
    elif isinstance(error, GRPCUnavailableError):
        return HTTPException(status_code=503, detail="Alarm service unavailable")
    elif isinstance(error, AlarmServiceError):
        return HTTPException(status_code=500, detail=str(error))
    else:
        logger.error(f"Unexpected error: {error}")
        return HTTPException(status_code=500, detail="Internal server error")


# API Endpoints
@router.post("/", response_model=AlarmResponse, status_code=201)
async def create_alarm(
    request: CreateAlarmRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new alarm
    
    Creates a new alarm in the system with the specified parameters.
    The alarm will be processed by the Rust Alarm Engine via gRPC.
    """
    try:
        # Convert metadata
        metadata_dict = metadata_to_dict(request.metadata)
        
        # Add user context
        metadata_dict['created_by'] = current_user.id
        metadata_dict['user_role'] = current_user.role
        
        # Create alarm via gRPC
        alarm = await alarm_client.create_alarm(
            patient_id=request.patient_id,
            alarm_type=request.alarm_type,
            severity=request.severity,
            message=request.message,
            description=request.description,
            metadata=metadata_dict,
            source=request.source or "fastapi",
            device_id=request.device_id
        )
        
        # Log alarm creation
        background_tasks.add_task(
            logger.info,
            f"Alarm created: {alarm.id} for patient {request.patient_id} by user {current_user.id}"
        )
        
        return alarm_to_response(alarm)
        
    except Exception as e:
        raise handle_grpc_error(e)


@router.get("/", response_model=List[AlarmResponse])
async def get_alarms(
    query: GetAlarmsQuery = Depends(),
    current_user: User = Depends(get_current_user)
):
    """
    Get alarms with filtering
    
    Retrieves alarms from the system with optional filtering by patient, status, severity, and type.
    Supports pagination with limit and offset parameters.
    """
    try:
        # Convert string parameters to enum values
        status_filter = None
        if query.status:
            try:
                status_filter = [alarm_pb2.AlarmStatus.Value(f"ALARM_STATUS_{query.status.upper()}")]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {query.status}")
        
        severity_filter = None
        if query.severity:
            try:
                severity_filter = [alarm_pb2.AlarmSeverity.Value(f"ALARM_SEVERITY_{query.severity.upper()}")]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {query.severity}")
        
        type_filter = None
        if query.alarm_type:
            try:
                type_filter = [alarm_pb2.AlarmType.Value(f"ALARM_TYPE_{query.alarm_type.upper()}")]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid alarm type: {query.alarm_type}")
        
        # Get alarms via gRPC
        alarms = await alarm_client.get_alarms(
            patient_id=query.patient_id,
            status=status_filter,
            severity=severity_filter,
            alarm_type=type_filter,
            limit=query.limit,
            offset=query.offset
        )
        
        return [alarm_to_response(alarm) for alarm in alarms]
        
    except HTTPException:
        raise
    except Exception as e:
        raise handle_grpc_error(e)


@router.get("/{alarm_id}", response_model=AlarmResponse)
async def get_alarm(
    alarm_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific alarm by ID
    
    Retrieves a specific alarm from the system using its unique identifier.
    """
    try:
        # Get all alarms and filter by ID (since there's no GetAlarmById method)
        alarms = await alarm_client.get_alarms(limit=1000)
        
        for alarm in alarms:
            if alarm.id == alarm_id:
                return alarm_to_response(alarm)
        
        raise HTTPException(status_code=404, detail=f"Alarm {alarm_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise handle_grpc_error(e)


@router.post("/{alarm_id}/acknowledge", response_model=AlarmResponse)
async def acknowledge_alarm(
    alarm_id: str,
    request: AcknowledgeAlarmRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Acknowledge an alarm
    
    Acknowledges an alarm to indicate that it has been seen and is being handled.
    """
    try:
        # Acknowledge alarm via gRPC
        alarm = await alarm_client.acknowledge_alarm(
            alarm_id=alarm_id,
            acknowledged_by=current_user.id,
            notes=request.notes
        )
        
        # Log acknowledgment
        background_tasks.add_task(
            logger.info,
            f"Alarm acknowledged: {alarm_id} by user {current_user.id}"
        )
        
        return alarm_to_response(alarm)
        
    except Exception as e:
        raise handle_grpc_error(e)


@router.post("/{alarm_id}/resolve", response_model=AlarmResponse)
async def resolve_alarm(
    alarm_id: str,
    request: ResolveAlarmRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Resolve an alarm
    
    Resolves an alarm to indicate that the underlying issue has been addressed.
    """
    try:
        # Resolve alarm via gRPC
        alarm = await alarm_client.resolve_alarm(
            alarm_id=alarm_id,
            resolved_by=current_user.id,
            resolution_notes=request.resolution_notes
        )
        
        # Log resolution
        background_tasks.add_task(
            logger.info,
            f"Alarm resolved: {alarm_id} by user {current_user.id}"
        )
        
        return alarm_to_response(alarm)
        
    except Exception as e:
        raise handle_grpc_error(e)


@router.get("/{patient_id}/stream")
async def stream_alarms(
    patient_id: str,
    severity: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Stream alarms for a patient in real-time
    
    Provides a server-sent events stream of real-time alarms for a specific patient.
    """
    try:
        # Convert severity to enum
        severity_filter = None
        if severity:
            try:
                severity_value = alarm_pb2.AlarmSeverity.Value(f"ALARM_SEVERITY_{severity.upper()}")
                severity_filter = [s for s in alarm_pb2.AlarmSeverity.values() if s >= severity_value]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        async def event_stream():
            """Generate SSE events for alarm stream"""
            try:
                async for alarm in alarm_client.stream_alarms(
                    patient_id=patient_id,
                    min_severity=severity_filter,
                    status_filter=[alarm_pb2.AlarmStatus.ALARM_STATUS_ACTIVE],
                    include_resolved=False
                ):
                    # Convert alarm to JSON
                    alarm_response = alarm_to_response(alarm)
                    alarm_json = json.dumps(alarm_response.dict(), default=str)
                    
                    # Send SSE event
                    yield f"data: {alarm_json}\n\n"
                    
            except Exception as e:
                logger.error(f"Stream error for patient {patient_id}: {e}")
                error_data = {"error": str(e), "timestamp": datetime.now().isoformat()}
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering for nginx
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise handle_grpc_error(e)


@router.get("/health/check")
async def health_check():
    """
    Health check endpoint
    
    Checks if the gRPC alarm service is healthy and accessible.
    """
    try:
        is_healthy = await alarm_client.health_check()
        
        if is_healthy:
            return {"status": "healthy", "service": "alarm-grpc-client"}
        else:
            return {"status": "unhealthy", "service": "alarm-grpc-client"}
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "service": "alarm-grpc-client", "error": str(e)}


@router.get("/metrics")
async def get_metrics():
    """
    Get gRPC client metrics
    
    Returns performance metrics and statistics for the gRPC client.
    """
    try:
        circuit_breaker_stats = alarm_client.get_circuit_breaker_stats()
        metrics_summary = alarm_client.get_metrics_summary()
        
        return {
            "circuit_breaker": circuit_breaker_stats,
            "metrics": metrics_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


# Lifecycle management
async def startup_alarm_client():
    """Initialize alarm client on startup"""
    global alarm_client
    alarm_client = AlarmGRPCClient()
    await alarm_client.connect()
    logger.info("Alarm gRPC client initialized")


async def shutdown_alarm_client():
    """Cleanup alarm client on shutdown"""
    global alarm_client
    if alarm_client:
        await alarm_client.disconnect()
        logger.info("Alarm gRPC client shutdown")


# FastAPI lifespan integration
async def register_alarm_routes(app):
    """Register alarm routes and lifespan events"""
    app.include_router(router)
    
    @app.on_event("startup")
    async def startup_event():
        await startup_alarm_client()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        await shutdown_alarm_client()
