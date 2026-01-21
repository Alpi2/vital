"""
Compliance Dashboard API Endpoints

REST API endpoints for the unified compliance dashboard with RBAC protection,
rate limiting, and comprehensive audit logging.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
import redis.asyncio as redis

from ..database import get_async_session
from ..services.unified_compliance_service import UnifiedComplianceService
from ..services.digital_signature_service import DigitalSignatureService
from ..services.report_generation_service import ReportGenerationService
from ..models.compliance_snapshot import ComplianceSnapshot, ComplianceAlert, ComplianceReport, ComplianceGap
from ..audit.audit_service import AuditService
from ..auth.rbac import RBACManager, get_current_user, require_permission
from ..auth.models import User
from ..config import get_settings
from ..tasks.compliance_tasks import generate_compliance_snapshot, generate_daily_reports

logger = logging.getLogger(__name__)

# Initialize router and security
router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])
security = HTTPBearer()
settings = get_settings()

# Initialize Redis for rate limiting
redis_client = None

async def get_redis_client():
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(settings.REDIS_URL)
    return redis_client

# Pydantic models for API requests/responses
class ComplianceScoreResponse(BaseModel):
    hipaa: float
    gdpr: float
    fda: float
    overall: float
    last_updated: datetime
    trends: Dict[str, float]

class ComplianceTrendRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365)
    domains: Optional[List[str]] = Field(default=None)

class ComplianceTrendResponse(BaseModel):
    trends: List[Dict[str, Any]]
    period: Dict[str, Any]

class ComplianceAlertResponse(BaseModel):
    id: str
    type: str
    domain: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool
    recommendations: List[str]

class ComplianceGapResponse(BaseModel):
    id: str
    domain: str
    category: str
    severity: str
    description: str
    recommendations: List[str]
    estimated_effort: str

class RegulatoryEventResponse(BaseModel):
    id: str
    title: str
    type: str
    domain: str
    due_date: datetime
    status: str
    description: str

class ReportGenerationRequest(BaseModel):
    report_type: str = Field(..., description="Type of report to generate")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_trends: bool = True
    include_alerts: bool = True
    include_gaps: bool = True
    format: str = Field(default="pdf", regex="^(pdf|excel|json|xml)$")
    sign_report: bool = True

class ReportGenerationResponse(BaseModel):
    report_id: str
    download_url: str
    expires_at: datetime
    size_bytes: int
    signed: bool

class SignatureStatusResponse(BaseModel):
    certificate_valid: bool
    certificate_info: Dict[str, Any]
    expiry_days: Optional[int]
    last_signature: Optional[Dict[str, Any]]

# Rate limiting decorator
async def rate_limit_check(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Rate limiting: 100 requests per minute per user"""
    user_id = credentials.credentials  # In production, extract actual user ID from token
    
    key = f"rate_limit:compliance:{user_id}"
    current_time = int(datetime.now().timestamp())
    
    # Use sliding window rate limiting
    await redis_client.zremrangebyscore(key, 0, current_time - 60)  # Remove old entries
    
    request_count = await redis_client.zcard(key)
    
    if request_count >= 100:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    await redis_client.zadd(key, {str(current_time): current_time})
    await redis_client.expire(key, 60)

# API Endpoints

@router.get("/dashboard", response_model=ComplianceScoreResponse)
async def get_dashboard_data(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    rate_limit: None = Depends(rate_limit_check),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get compliance dashboard data including scores and trends
    
    Requires: compliance.view permission
    Rate limit: 100 requests/minute
    """
    try:
        # Check permissions
        await require_permission(current_user, "compliance.view")
        
        # Initialize services
        audit_service = AuditService(db)
        unified_service = UnifiedComplianceService(db, audit_service)
        
        # Get latest compliance snapshot
        stmt = select(ComplianceSnapshot).order_by(
            ComplianceSnapshot.created_at.desc()
        ).limit(1)
        
        result = await db.execute(stmt)
        snapshot = result.scalar_one_or_none()
        
        if not snapshot:
            # Generate snapshot if none exists
            snapshot = await unified_service.generate_compliance_snapshot()
        
        # Calculate trends (compare with previous snapshot)
        trends = await unified_service.calculate_trends(snapshot.id)
        
        response = ComplianceScoreResponse(
            hipaa=snapshot.hipaa_score,
            gdpr=snapshot.gdpr_score,
            fda=snapshot.fda_score,
            overall=snapshot.overall_score,
            last_updated=snapshot.created_at,
            trends=trends
        )
        
        # Log access
        await audit_service.log_event(
            action="dashboard_accessed",
            resource_type="compliance_dashboard",
            user_id=current_user.id,
            details={
                "snapshot_id": snapshot.id,
                "access_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")

@router.get("/trends", response_model=ComplianceTrendResponse)
async def get_compliance_trends(
    days: int = Query(default=30, ge=1, le=365),
    domains: Optional[str] = Query(default=None),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    rate_limit: None = Depends(rate_limit_check),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get compliance trends over specified period
    
    Requires: compliance.view permission
    """
    try:
        await require_permission(current_user, "compliance.view")
        
        audit_service = AuditService(db)
        unified_service = UnifiedComplianceService(db, audit_service)
        
        # Parse domains
        domain_list = domains.split(",") if domains else ["hipaa", "gdpr", "fda"]
        
        # Get trend data
        trends = await unified_service.get_compliance_trends(days, domain_list)
        
        response = ComplianceTrendResponse(
            trends=trends,
            period={
                "days": days,
                "start_date": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
                "end_date": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Log access
        await audit_service.log_event(
            action="trends_accessed",
            resource_type="compliance_trends",
            user_id=current_user.id,
            details={
                "days": days,
                "domains": domain_list
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get compliance trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trends data")

@router.get("/alerts", response_model=List[ComplianceAlertResponse])
async def get_compliance_alerts(
    severity: Optional[str] = Query(default=None, regex="^(critical|high|medium|low)$"),
    domain: Optional[str] = Query(default=None, regex="^(HIPAA|GDPR|FDA)$"),
    resolved: Optional[bool] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    rate_limit: None = Depends(rate_limit_check),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get compliance alerts with filtering options
    
    Requires: compliance.view permission
    """
    try:
        await require_permission(current_user, "compliance.view")
        
        # Build query
        stmt = select(ComplianceAlert)
        
        # Apply filters
        conditions = []
        if severity:
            conditions.append(ComplianceAlert.severity == severity)
        if domain:
            conditions.append(ComplianceAlert.domain == domain)
        if resolved is not None:
            conditions.append(ComplianceAlert.resolved == resolved)
        
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        # Order and paginate
        stmt = stmt.order_by(
            ComplianceAlert.created_at.desc()
        ).offset(offset).limit(limit)
        
        result = await db.execute(stmt)
        alerts = result.scalars().all()
        
        # Convert to response model
        response_alerts = []
        for alert in alerts:
            response_alerts.append(ComplianceAlertResponse(
                id=alert.id,
                type=alert.alert_type,
                domain=alert.domain,
                severity=alert.severity,
                message=alert.message,
                timestamp=alert.created_at,
                resolved=alert.resolved,
                recommendations=alert.recommendations or []
            ))
        
        # Log access
        await AuditService(db).log_event(
            action="alerts_accessed",
            resource_type="compliance_alerts",
            user_id=current_user.id,
            details={
                "filters": {
                    "severity": severity,
                    "domain": domain,
                    "resolved": resolved
                },
                "count": len(response_alerts)
            }
        )
        
        return response_alerts
        
    except Exception as e:
        logger.error(f"Failed to get compliance alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolution_note: str = Query(..., min_length=1, max_length=500),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Resolve a compliance alert
    
    Requires: compliance.manage permission
    """
    try:
        await require_permission(current_user, "compliance.manage")
        
        audit_service = AuditService(db)
        
        # Find alert
        stmt = select(ComplianceAlert).where(ComplianceAlert.id == alert_id)
        result = await db.execute(stmt)
        alert = result.scalar_one_or_none()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Update alert
        alert.resolved = True
        alert.resolved_at = datetime.now(timezone.utc)
        alert.resolved_by = current_user.id
        alert.resolution_note = resolution_note
        
        await db.commit()
        
        # Log resolution
        await audit_service.log_event(
            action="alert_resolved",
            resource_type="compliance_alert",
            resource_id=alert_id,
            user_id=current_user.id,
            details={
                "resolution_note": resolution_note,
                "resolved_at": alert.resolved_at.isoformat()
            }
        )
        
        return {"message": "Alert resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")

@router.get("/gaps", response_model=List[ComplianceGapResponse])
async def get_compliance_gaps(
    domain: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    rate_limit: None = Depends(rate_limit_check),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get compliance gaps with filtering options
    
    Requires: compliance.view permission
    """
    try:
        await require_permission(current_user, "compliance.view")
        
        # Build query
        stmt = select(ComplianceGap)
        
        # Apply filters
        conditions = []
        if domain:
            conditions.append(ComplianceGap.domain == domain)
        if severity:
            conditions.append(ComplianceGap.severity == severity)
        
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        stmt = stmt.order_by(
            ComplianceGap.severity.desc(),
            ComplianceGap.created_at.desc()
        )
        
        result = await db.execute(stmt)
        gaps = result.scalars().all()
        
        # Convert to response model
        response_gaps = []
        for gap in gaps:
            response_gaps.append(ComplianceGapResponse(
                id=gap.id,
                domain=gap.domain,
                category=gap.category,
                severity=gap.severity,
                description=gap.description,
                recommendations=gap.recommendations or [],
                estimated_effort=gap.estimated_effort or "Unknown"
            ))
        
        # Log access
        await AuditService(db).log_event(
            action="gaps_accessed",
            resource_type="compliance_gaps",
            user_id=current_user.id,
            details={
                "filters": {"domain": domain, "severity": severity},
                "count": len(response_gaps)
            }
        )
        
        return response_gaps
        
    except Exception as e:
        logger.error(f"Failed to get compliance gaps: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve gaps")

@router.get("/calendar", response_model=List[RegulatoryEventResponse])
async def get_regulatory_calendar(
    days_ahead: int = Query(default=30, ge=1, le=365),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    rate_limit: None = Depends(rate_limit_check),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get regulatory calendar events
    
    Requires: compliance.view permission
    """
    try:
        await require_permission(current_user, "compliance.view")
        
        audit_service = AuditService(db)
        unified_service = UnifiedComplianceService(db, audit_service)
        
        # Get calendar events
        events = await unified_service.get_regulatory_calendar(days_ahead)
        
        # Convert to response model
        response_events = []
        for event in events:
            response_events.append(RegulatoryEventResponse(
                id=event.id,
                title=event.title,
                type=event.event_type,
                domain=event.domain,
                due_date=event.due_date,
                status=event.status,
                description=event.description or ""
            ))
        
        # Log access
        await audit_service.log_event(
            action="calendar_accessed",
            resource_type="regulatory_calendar",
            user_id=current_user.id,
            details={"days_ahead": days_ahead}
        )
        
        return response_events
        
    except Exception as e:
        logger.error(f"Failed to get regulatory calendar: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve calendar")

@router.post("/reports/generate", response_model=ReportGenerationResponse)
async def generate_report(
    request: ReportGenerationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Generate compliance report
    
    Requires: compliance.report permission
    """
    try:
        await require_permission(current_user, "compliance.report")
        
        audit_service = AuditService(db)
        signature_service = DigitalSignatureService(db, audit_service)
        report_service = ReportGenerationService(db, audit_service, signature_service)
        
        # Set default date range if not provided
        if not request.start_date:
            request.start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if not request.end_date:
            request.end_date = datetime.now(timezone.utc)
        
        # Generate report based on type
        if request.report_type == "comprehensive":
            report = await report_service.generate_comprehensive_report(
                start_date=request.start_date,
                end_date=request.end_date,
                include_trends=request.include_trends,
                include_alerts=request.include_alerts,
                include_gaps=request.include_gaps
            )
        elif request.report_type == "executive":
            report = await report_service.generate_executive_summary(
                start_date=request.start_date,
                end_date=request.end_date
            )
        elif request.report_type == "fda-510k":
            report = await report_service.generate_fda_510k_package(
                submission_date=request.end_date
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid report type")
        
        # Generate download URL (implement secure download logic)
        download_url = f"/api/v1/compliance/reports/{report.id}/download"
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        
        # Log report generation
        await audit_service.log_event(
            action="report_generated",
            resource_type="compliance_report",
            resource_id=report.id,
            user_id=current_user.id,
            details={
                "report_type": request.report_type,
                "format": request.format,
                "signed": request.sign_report,
                "date_range": {
                    "start": request.start_date.isoformat(),
                    "end": request.end_date.isoformat()
                }
            }
        )
        
        response = ReportGenerationResponse(
            report_id=report.id,
            download_url=download_url,
            expires_at=expires_at,
            size_bytes=len(report.content) if hasattr(report, 'content') else 0,
            signed=request.sign_report
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Download generated compliance report
    
    Requires: compliance.report permission
    """
    try:
        await require_permission(current_user, "compliance.report")
        
        # Get report from database
        stmt = select(ComplianceReport).where(ComplianceReport.id == report_id)
        result = await db.execute(stmt)
        report = result.scalar_one_or_none()
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Check if report has expired
        if report.expires_at and report.expires_at < datetime.now(timezone.utc):
            raise HTTPException(status_code=410, detail="Report has expired")
        
        # Log download
        await AuditService(db).log_event(
            action="report_downloaded",
            resource_type="compliance_report",
            resource_id=report_id,
            user_id=current_user.id,
            details={
                "format": report.format,
                "download_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Return file
        media_type = {
            "pdf": "application/pdf",
            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "json": "application/json",
            "xml": "application/xml"
        }.get(report.format, "application/octet-stream")
        
        return StreamingResponse(
            BytesIO(report.content),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={report.filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download report: {e}")
        raise HTTPException(status_code=500, detail="Failed to download report")

@router.get("/signature/status", response_model=SignatureStatusResponse)
async def get_signature_status(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    rate_limit: None = Depends(rate_limit_check),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get digital signature service status
    
    Requires: compliance.view permission
    """
    try:
        await require_permission(current_user, "compliance.view")
        
        audit_service = AuditService(db)
        signature_service = DigitalSignatureService(db, audit_service)
        
        # Get certificate status
        cert_report = await signature_service.generate_certificate_report()
        
        # Get last signature
        stmt = select(ComplianceReport).where(
            ComplianceReport.digital_signature == True
        ).order_by(ComplianceReport.signed_at.desc()).limit(1)
        
        result = await db.execute(stmt)
        last_signed_report = result.scalar_one_or_none()
        
        last_signature = None
        if last_signed_report:
            last_signature = {
                "report_id": last_signed_report.id,
                "signed_at": last_signed_report.signed_at.isoformat(),
                "signer": last_signed_report.signature_metadata.get("signer_name") if last_signed_report.signature_metadata else None
            }
        
        response = SignatureStatusResponse(
            certificate_valid=cert_report["certificate_valid"],
            certificate_info=cert_report["certificate_info"],
            expiry_days=cert_report["expiry_days"],
            last_signature=last_signature
        )
        
        # Log access
        await audit_service.log_event(
            action="signature_status_accessed",
            resource_type="digital_signature",
            user_id=current_user.id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get signature status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signature status")

@router.post("/snapshot/generate")
async def generate_snapshot(
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Trigger compliance snapshot generation
    
    Requires: compliance.manage permission
    """
    try:
        await require_permission(current_user, "compliance.manage")
        
        # Queue background task
        task = generate_compliance_snapshot.delay()
        
        # Log task creation
        await AuditService(db).log_event(
            action="snapshot_generation_triggered",
            resource_type="compliance_snapshot",
            user_id=current_user.id,
            details={
                "task_id": task.id,
                "triggered_by": current_user.id,
                "triggered_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
        return {
            "message": "Compliance snapshot generation started",
            "task_id": task.id
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger snapshot generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger snapshot generation")

@router.get("/tasks/{task_id}/status")
async def get_task_status(
    task_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: User = Depends(get_current_user)
):
    """
    Get status of a background task
    
    Requires: compliance.view permission
    """
    try:
        await require_permission(current_user, "compliance.view")
        
        from ..tasks.compliance_tasks import get_task_status
        
        status = get_task_status(task_id)
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for compliance service"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "compliance_dashboard",
        "version": "1.0.0"
    }
