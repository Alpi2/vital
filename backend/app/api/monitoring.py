"""
GDPR Real-time Monitoring Dashboard API
Provides real-time compliance metrics and monitoring
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
from pydantic import BaseModel, Field
from datetime import datetime, timezone, timedelta
import asyncio
import uuid
import json

from app.database import get_db
from app.models.gdpr import (
    PatientConsent, DataSubjectRequest, RequestStatus, DataBreachRecord,
    ConsentAuditLog, DPIARecord, ROPARecord
)
from app.models.patient import Patient
from app.models.user import User
from app.security import get_current_user, require_admin
from app.tasks.gdpr_tasks import GDPRBackgroundTasks
from app.celery_app import celery_app

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


# Pydantic Models
class ComplianceMetrics(BaseModel):
    """GDPR compliance metrics"""
    timestamp: datetime
    consent_compliance: float = Field(..., ge=0, le=100)
    request_processing: float = Field(..., ge=0, le=100)
    data_retention: float = Field(..., ge=0, le=100)
    security_compliance: float = Field(..., ge=0, le=100)
    overall_score: float = Field(..., ge=0, le=100)
    trend_7_days: float
    trend_30_days: float


class RequestMetrics(BaseModel):
    """Data subject request metrics"""
    total_requests: int
    pending_requests: int
    in_progress_requests: int
    completed_requests: int
    rejected_requests: int
    overdue_requests: int
    average_processing_time: float
    requests_by_type: Dict[str, int]
    requests_by_status: Dict[str, int]


class ConsentMetrics(BaseModel):
    """Consent management metrics"""
    total_consents: int
    active_consents: int
    revoked_consents: int
    expired_consents: int
    consents_by_type: Dict[str, int]
    consent_grant_rate: float
    consent_withdrawal_rate: float


class SecurityMetrics(BaseModel):
    """Security and breach metrics"""
    active_breaches: int
    resolved_breaches: int
    high_risk_breaches: int
    breach_notifications_sent: int
    average_breach_resolution_time: float
    security_incidents_24h: int
    failed_login_attempts_24h: int


class SystemMetrics(BaseModel):
    """System performance metrics"""
    api_response_time: float
    database_connections: int
    queue_length: int
    storage_usage: Dict[str, Any]
    error_rate: float
    uptime_percentage: float


class AlertInfo(BaseModel):
    """Alert information"""
    id: str
    type: str
    severity: str
    title: str
    message: str
    timestamp: datetime
    resolved: bool
    metadata: Dict[str, Any]


class DashboardData(BaseModel):
    """Complete dashboard data"""
    compliance_metrics: ComplianceMetrics
    request_metrics: RequestMetrics
    consent_metrics: ConsentMetrics
    security_metrics: SecurityMetrics
    system_metrics: SystemMetrics
    recent_alerts: List[AlertInfo]
    last_updated: datetime


@router.get("/dashboard", response_model=DashboardData)
async def get_dashboard_data(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get complete GDPR monitoring dashboard data"""
    try:
        # Get all metrics concurrently
        compliance_metrics_task = get_compliance_metrics(db)
        request_metrics_task = get_request_metrics(db)
        consent_metrics_task = get_consent_metrics(db)
        security_metrics_task = get_security_metrics(db)
        system_metrics_task = get_system_metrics()
        recent_alerts_task = get_recent_alerts(db)
        
        # Wait for all tasks to complete
        compliance_metrics, request_metrics, consent_metrics, security_metrics, system_metrics, recent_alerts = await asyncio.gather(
            compliance_metrics_task,
            request_metrics_task,
            consent_metrics_task,
            security_metrics_task,
            system_metrics_task,
            recent_alerts_task
        )
        
        return DashboardData(
            compliance_metrics=compliance_metrics,
            request_metrics=request_metrics,
            consent_metrics=consent_metrics,
            security_metrics=security_metrics,
            system_metrics=system_metrics,
            recent_alerts=recent_alerts,
            last_updated=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@router.get("/compliance", response_model=ComplianceMetrics)
async def get_compliance_metrics_endpoint(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get GDPR compliance metrics"""
    return await get_compliance_metrics(db)


@router.get("/requests", response_model=RequestMetrics)
async def get_request_metrics_endpoint(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get data subject request metrics"""
    return await get_request_metrics(db)


@router.get("/consents", response_model=ConsentMetrics)
async def get_consent_metrics_endpoint(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get consent management metrics"""
    return await get_consent_metrics(db)


@router.get("/security", response_model=SecurityMetrics)
async def get_security_metrics_endpoint(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get security and breach metrics"""
    return await get_security_metrics(db)


@router.get("/system", response_model=SystemMetrics)
async def get_system_metrics_endpoint(
    current_user: User = Depends(require_admin)
):
    """Get system performance metrics"""
    return await get_system_metrics()


@router.get("/alerts", response_model=List[AlertInfo])
async def get_alerts_endpoint(
    limit: int = Query(50, ge=1, le=100),
    severity: Optional[str] = Query(None),
    resolved: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get monitoring alerts"""
    return await get_recent_alerts(db, limit, severity, resolved)


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Resolve an alert"""
    try:
        # This would update the alert in your alert system
        # Implementation depends on your alert storage
        return {"success": True, "message": "Alert resolved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")


@router.get("/real-time")
async def get_real_time_updates(
    current_user: User = Depends(require_admin)
):
    """Get real-time monitoring updates (WebSocket endpoint would be better)"""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "Use WebSocket endpoint for real-time updates",
        "websocket_url": "/ws/monitoring"
    }


# Helper functions

async def get_compliance_metrics(db: AsyncSession) -> ComplianceMetrics:
    """Calculate GDPR compliance metrics"""
    
    # Consent compliance
    total_patients_query = select(func.count(Patient.id))
    total_patients_result = await db.execute(total_patients_query)
    total_patients = total_patients_result.scalar() or 0
    
    active_consents_query = select(func.count(PatientConsent.id)).where(
        and_(
            PatientConsent.granted == True,
            PatientConsent.revoked_at.is_(None),
            or_(
                PatientConsent.expiry_date.is_(None),
                PatientConsent.expiry_date > datetime.now(timezone.utc)
            )
        )
    )
    active_consents_result = await db.execute(active_consents_query)
    active_consents = active_consents_result.scalar() or 0
    
    consent_compliance = (active_consents / max(total_patients, 1)) * 100
    
    # Request processing compliance
    total_requests_query = select(func.count(DataSubjectRequest.id))
    total_requests_result = await db.execute(total_requests_query)
    total_requests = total_requests_result.scalar() or 0
    
    completed_requests_query = select(func.count(DataSubjectRequest.id)).where(
        DataSubjectRequest.status == RequestStatus.COMPLETED
    )
    completed_requests_result = await db.execute(completed_requests_query)
    completed_requests = completed_requests_result.scalar() or 0
    
    overdue_requests_query = select(func.count(DataSubjectRequest.id)).where(
        and_(
            DataSubjectRequest.status.in_([RequestStatus.PENDING, RequestStatus.IN_PROGRESS]),
            DataSubjectRequest.deadline < datetime.now(timezone.utc)
        )
    )
    overdue_requests_result = await db.execute(overdue_requests_query)
    overdue_requests = overdue_requests_result.scalar() or 0
    
    request_processing = ((total_requests - overdue_requests) / max(total_requests, 1)) * 100
    
    # Data retention compliance (placeholder)
    data_retention = 100.0  # Would check actual retention compliance
    
    # Security compliance (placeholder)
    security_compliance = 100.0  # Would check actual security measures
    
    # Overall score
    overall_score = (
        consent_compliance * 0.3 +
        request_processing * 0.3 +
        data_retention * 0.2 +
        security_compliance * 0.2
    )
    
    # Calculate trends (simplified)
    trend_7_days = 0.0  # Would compare with 7 days ago
    trend_30_days = 0.0  # Would compare with 30 days ago
    
    return ComplianceMetrics(
        timestamp=datetime.now(timezone.utc),
        consent_compliance=consent_compliance,
        request_processing=request_processing,
        data_retention=data_retention,
        security_compliance=security_compliance,
        overall_score=overall_score,
        trend_7_days=trend_7_days,
        trend_30_days=trend_30_days
    )


async def get_request_metrics(db: AsyncSession) -> RequestMetrics:
    """Get data subject request metrics"""
    
    # Total requests by status
    total_requests_query = select(func.count(DataSubjectRequest.id))
    total_requests_result = await db.execute(total_requests_query)
    total_requests = total_requests_result.scalar() or 0
    
    pending_requests_query = select(func.count(DataSubjectRequest.id)).where(
        DataSubjectRequest.status == RequestStatus.PENDING
    )
    pending_requests_result = await db.execute(pending_requests_query)
    pending_requests = pending_requests_result.scalar() or 0
    
    in_progress_requests_query = select(func.count(DataSubjectRequest.id)).where(
        DataSubjectRequest.status == RequestStatus.IN_PROGRESS
    )
    in_progress_requests_result = await db.execute(in_progress_requests_query)
    in_progress_requests = in_progress_requests_result.scalar() or 0
    
    completed_requests_query = select(func.count(DataSubjectRequest.id)).where(
        DataSubjectRequest.status == RequestStatus.COMPLETED
    )
    completed_requests_result = await db.execute(completed_requests_query)
    completed_requests = completed_requests_result.scalar() or 0
    
    rejected_requests_query = select(func.count(DataSubjectRequest.id)).where(
        DataSubjectRequest.status == RequestStatus.REJECTED
    )
    rejected_requests_result = await db.execute(rejected_requests_query)
    rejected_requests = rejected_requests_result.scalar() or 0
    
    overdue_requests_query = select(func.count(DataSubjectRequest.id)).where(
        and_(
            DataSubjectRequest.status.in_([RequestStatus.PENDING, RequestStatus.IN_PROGRESS]),
            DataSubjectRequest.deadline < datetime.now(timezone.utc)
        )
    )
    overdue_requests_result = await db.execute(overdue_requests_query)
    overdue_requests = overdue_requests_result.scalar() or 0
    
    # Requests by type
    requests_by_type_query = select(
        DataSubjectRequest.request_type,
        func.count(DataSubjectRequest.id)
    ).group_by(DataSubjectRequest.request_type)
    
    requests_by_type_result = await db.execute(requests_by_type_query)
    requests_by_type = {row[0].value: row[1] for row in requests_by_type_result}
    
    # Average processing time
    avg_processing_time_query = select(
        func.avg(
            func.extract('epoch', DataSubjectRequest.completed_date - DataSubjectRequest.request_date)
        )
    ).where(
        DataSubjectRequest.status == RequestStatus.COMPLETED
    )
    
    avg_processing_time_result = await db.execute(avg_processing_time_query)
    average_processing_time = avg_processing_time_result.scalar() or 0
    
    return RequestMetrics(
        total_requests=total_requests,
        pending_requests=pending_requests,
        in_progress_requests=in_progress_requests,
        completed_requests=completed_requests,
        rejected_requests=rejected_requests,
        overdue_requests=overdue_requests,
        average_processing_time=average_processing_time,
        requests_by_type=requests_by_type,
        requests_by_status={
            "pending": pending_requests,
            "in_progress": in_progress_requests,
            "completed": completed_requests,
            "rejected": rejected_requests
        }
    )


async def get_consent_metrics(db: AsyncSession) -> ConsentMetrics:
    """Get consent management metrics"""
    
    # Total consents
    total_consents_query = select(func.count(PatientConsent.id))
    total_consents_result = await db.execute(total_consents_query)
    total_consents = total_consents_result.scalar() or 0
    
    # Active consents
    active_consents_query = select(func.count(PatientConsent.id)).where(
        and_(
            PatientConsent.granted == True,
            PatientConsent.revoked_at.is_(None),
            or_(
                PatientConsent.expiry_date.is_(None),
                PatientConsent.expiry_date > datetime.now(timezone.utc)
            )
        )
    )
    active_consents_result = await db.execute(active_consents_query)
    active_consents = active_consents_result.scalar() or 0
    
    # Revoked consents
    revoked_consents_query = select(func.count(PatientConsent.id)).where(
        PatientConsent.revoked_at.is_not(None)
    )
    revoked_consents_result = await db.execute(revoked_consents_query)
    revoked_consents = revoked_consents_result.scalar() or 0
    
    # Expired consents
    expired_consents_query = select(func.count(PatientConsent.id)).where(
        and_(
            PatientConsent.expiry_date.is_not(None),
            PatientConsent.expiry_date < datetime.now(timezone.utc)
        )
    )
    expired_consents_result = await db.execute(expired_consents_query)
    expired_consents = expired_consents_result.scalar() or 0
    
    # Consents by type
    consents_by_type_query = select(
        PatientConsent.consent_type,
        func.count(PatientConsent.id)
    ).group_by(PatientConsent.consent_type)
    
    consents_by_type_result = await db.execute(consents_by_type_query)
    consents_by_type = {row[0].value: row[1] for row in consents_by_type_result}
    
    # Consent grant rate (last 30 days)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    granted_consents_query = select(func.count(PatientConsent.id)).where(
        PatientConsent.granted_at >= thirty_days_ago
    )
    granted_consents_result = await db.execute(granted_consents_query)
    granted_consents = granted_consents_result.scalar() or 0
    
    consent_grant_rate = granted_consents / max(1, total_consents) * 100
    
    # Consent withdrawal rate (last 30 days)
    withdrawn_consents_query = select(func.count(PatientConsent.id)).where(
        PatientConsent.revoked_at >= thirty_days_ago
    )
    withdrawn_consents_result = await db.execute(withdrawn_consents_query)
    withdrawn_consents = withdrawn_consents_result.scalar() or 0
    
    consent_withdrawal_rate = withdrawn_consents / max(1, total_consents) * 100
    
    return ConsentMetrics(
        total_consents=total_consents,
        active_consents=active_consents,
        revoked_consents=revoked_consents,
        expired_consents=expired_consents,
        consents_by_type=consents_by_type,
        consent_grant_rate=consent_grant_rate,
        consent_withdrawal_rate=consent_withdrawal_rate
    )


async def get_security_metrics(db: AsyncSession) -> SecurityMetrics:
    """Get security and breach metrics"""
    
    # Active breaches
    active_breaches_query = select(func.count(DataBreachRecord.id)).where(
        DataBreachRecord.status.in_(['investigating', 'contained'])
    )
    active_breaches_result = await db.execute(active_breaches_query)
    active_breaches = active_breaches_result.scalar() or 0
    
    # Resolved breaches
    resolved_breaches_query = select(func.count(DataBreachRecord.id)).where(
        DataBreachRecord.status == 'resolved'
    )
    resolved_breaches_result = await db.execute(resolved_breaches_query)
    resolved_breaches = resolved_breaches_result.scalar() or 0
    
    # High risk breaches
    high_risk_breaches_query = select(func.count(DataBreachRecord.id)).where(
        DataBreachRecord.high_risk == True
    )
    high_risk_breaches_result = await db.execute(high_risk_breaches_query)
    high_risk_breaches = high_risk_breaches_result.scalar() or 0
    
    # Breach notifications sent
    notifications_sent_query = select(func.count(DataBreachRecord.id)).where(
        DataBreachRecord.authority_notified.is_not(None)
    )
    notifications_sent_result = await db.execute(notifications_sent_query)
    breach_notifications_sent = notifications_sent_result.scalar() or 0
    
    # Average breach resolution time
    avg_resolution_time_query = select(
        func.avg(
            func.extract('epoch', DataBreachRecord.assessment_completed - DataBreachRecord.breach_discovered)
        )
    ).where(
        DataBreachRecord.status == 'resolved'
    )
    
    avg_resolution_time_result = await db.execute(avg_resolution_time_query)
    average_breach_resolution_time = avg_resolution_time_result.scalar() or 0
    
    # Security incidents (placeholder - would come from security logs)
    security_incidents_24h = 0
    
    # Failed login attempts (placeholder - would come from auth logs)
    failed_login_attempts_24h = 0
    
    return SecurityMetrics(
        active_breaches=active_breaches,
        resolved_breaches=resolved_breaches,
        high_risk_breaches=high_risk_breaches,
        breach_notifications_sent=breach_notifications_sent,
        average_breach_resolution_time=average_breach_resolution_time,
        security_incidents_24h=security_incidents_24h,
        failed_login_attempts_24h=failed_login_attempts_24h
    )


async def get_system_metrics() -> SystemMetrics:
    """Get system performance metrics"""
    
    # API response time (placeholder - would come from monitoring)
    api_response_time = 0.5
    
    # Database connections
    try:
        # Get Celery stats for queue length
        stats = celery_app.control.inspect().stats()
        queue_length = sum(worker.get('pool'].get('max-concurrency', 0) for worker in stats.values()) if stats else 0
    except:
        queue_length = 0
    
    # Storage usage (placeholder - would check actual storage)
    storage_usage = {
        "database": "10.5 GB",
        "logs": "2.3 GB",
        "exports": "1.8 GB",
        "total": "14.6 GB"
    }
    
    # Error rate (placeholder - would come from error tracking)
    error_rate = 0.02
    
    # Uptime (placeholder - would come from monitoring)
    uptime_percentage = 99.9
    
    return SystemMetrics(
        api_response_time=api_response_time,
        database_connections=queue_length,
        queue_length=queue_length,
        storage_usage=storage_usage,
        error_rate=error_rate,
        uptime_percentage=uptime_percentage
    )


async def get_recent_alerts(
    db: AsyncSession, 
    limit: int = 50, 
    severity: Optional[str] = None, 
    resolved: Optional[bool] = None
) -> List[AlertInfo]:
    """Get recent monitoring alerts"""
    
    # This is a placeholder implementation
    # In practice, you would have an alerts table or external monitoring system
    
    alerts = []
    
    # Sample alerts based on current metrics
    compliance_metrics = await get_compliance_metrics(db)
    
    if compliance_metrics.overall_score < 90:
        alerts.append(AlertInfo(
            id=str(uuid.uuid4()),
            type="compliance",
            severity="warning",
            title="Low Compliance Score",
            message=f"Overall compliance score is {compliance_metrics.overall_score:.1f}%",
            timestamp=datetime.now(timezone.utc),
            resolved=False,
            metadata={"score": compliance_metrics.overall_score}
        ))
    
    request_metrics = await get_request_metrics(db)
    
    if request_metrics.overdue_requests > 0:
        alerts.append(AlertInfo(
            id=str(uuid.uuid4()),
            type="requests",
            severity="error",
            title="Overdue Requests",
            message=f"{request_metrics.overdue_requests} requests are overdue",
            timestamp=datetime.now(timezone.utc),
            resolved=False,
            metadata={"overdue_count": request_metrics.overdue_requests}
        ))
    
    security_metrics = await get_security_metrics(db)
    
    if security_metrics.active_breaches > 0:
        alerts.append(AlertInfo(
            id=str(uuid.uuid4()),
            type="security",
            severity="critical",
            title="Active Security Breaches",
            message=f"{security_metrics.active_breaches} security breaches are currently active",
            timestamp=datetime.now(timezone.utc),
            resolved=False,
            metadata={"active_breaches": security_metrics.active_breaches}
        ))
    
    # Filter by severity if specified
    if severity:
        alerts = [alert for alert in alerts if alert.severity == severity]
    
    # Filter by resolved status if specified
    if resolved is not None:
        alerts = [alert for alert in alerts if alert.resolved == resolved]
    
    # Sort by timestamp and limit
    alerts.sort(key=lambda x: x.timestamp, reverse=True)
    return alerts[:limit]


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }


@router.get("/metrics/prometheus")
async def prometheus_metrics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Export metrics in Prometheus format"""
    
    # Get all metrics
    compliance_metrics = await get_compliance_metrics(db)
    request_metrics = await get_request_metrics(db)
    consent_metrics = await get_consent_metrics(db)
    security_metrics = await get_security_metrics(db)
    
    # Format as Prometheus metrics
    prometheus_metrics = f"""
# HELP gdpr_compliance_score GDPR compliance score percentage
# TYPE gdpr_compliance_score gauge
gdpr_compliance_score {compliance_metrics.overall_score}

# HELP gdpr_requests_total Total number of data subject requests
# TYPE gdpr_requests_total counter
gdpr_requests_total {request_metrics.total_requests}

# HELP gdpr_requests_pending Number of pending data subject requests
# TYPE gdpr_requests_pending gauge
gdpr_requests_pending {request_metrics.pending_requests}

# HELP gdpr_consents_active Number of active consents
# TYPE gdpr_consents_active gauge
gdpr_consents_active {consent_metrics.active_consents}

# HELP gdpr_breaches_active Number of active security breaches
# TYPE gdpr_breaches_active gauge
gdpr_breaches_active {security_metrics.active_breaches}
"""
    
    return Response(
        content=prometheus_metrics,
        media_type="text/plain"
    )
