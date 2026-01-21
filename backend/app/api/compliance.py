"""
Compliance API Endpoints
HIPAA compliance reporting and monitoring with proper dependency injection
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import datetime
import io
import csv
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from app.database import get_db
from app.security.dependencies import get_current_user, require_role
from app.models.user import User
from app.services.audit_service import AuditService
from app.compliance.hipaa_validator import HIPAAValidator, ComplianceReport, ComplianceStatus
from app.compliance.background_tasks import task_manager

router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])


async def get_audit_service() -> AuditService:
    """Dependency injection for audit service"""
    return AuditService()


@router.get("/hipaa/report")
async def get_hipaa_compliance_report(
    background_tasks: BackgroundTasks,
    force_refresh: bool = Query(False, description="Force new report generation"),
    priority: str = Query("normal", regex="^(normal|high)$", description="Task priority"),
    current_user: User = Depends(require_role(["super_admin", "admin"])),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends(get_audit_service)
):
    """
    Generate or retrieve HIPAA compliance report
    
    Uses background tasks to avoid timeout issues. Returns cached report if available
    or submits a new generation task.
    """
    
    # Try to get cached report first (unless force refresh)
    if not force_refresh:
        cached_report = await task_manager.get_cached_report(max_age_minutes=30)
        if cached_report:
            await audit_service.log(
                action="compliance.report_accessed_cached",
                user_id=str(current_user.id),
                resource_type="compliance_report",
                description="Accessed cached HIPAA compliance report",
                is_phi_access=False,
                db=db
            )
            
            return {
                "report_id": cached_report.report_id,
                "cached": True,
                **cached_report.__dict__
            }
    
    # Submit background task for report generation
    task_id = await task_manager.submit_compliance_report_task(
        user_id=str(current_user.id),
        priority=priority
    )
    
    await audit_service.log(
        action="compliance.report_requested",
        user_id=str(current_user.id),
        resource_type="compliance_task",
        resource_id=task_id,
        description=f"Submitted HIPAA compliance report generation task",
        request_data={"priority": priority, "force_refresh": force_refresh},
        is_phi_access=False,
        db=db
    )
    
    return {
        "message": "Compliance report generation started",
        "task_id": task_id,
        "status": "queued",
        "estimated_time": "2-5 minutes",
        "check_status_url": f"/api/v1/compliance/task/{task_id}"
    }


@router.get("/hipaa/report/{task_id}")
async def get_compliance_report_result(
    task_id: str,
    current_user: User = Depends(require_role(["super_admin", "admin"])),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends(get_audit_service)
):
    """Get the result of a compliance report generation task"""
    
    task_status = await task_manager.get_task_status(task_id)
    
    if task_status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    
    await audit_service.log(
        action="compliance.task_status_checked",
        user_id=str(current_user.id),
        resource_type="compliance_task",
        resource_id=task_id,
        description=f"Checked compliance task status: {task_status['status']}",
        is_phi_access=False,
        db=db
    )
    
    return task_status


@router.get("/hipaa/dashboard")
async def get_compliance_dashboard(
    current_user: User = Depends(require_role(["super_admin", "admin"])),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends(get_audit_service)
):
    """Get real-time compliance dashboard data"""
    
    # Try to get cached report for dashboard
    cached_report = await task_manager.get_cached_report(max_age_minutes=5)
    
    if not cached_report:
        # If no cached report, try to get latest completed task
        task_history = await task_manager.get_task_history(limit=10)
        latest_completed = None
        
        for task in task_history:
            if task.get("status") == "completed" and task.get("result"):
                latest_completed = task
                break
        
        if latest_completed:
            cached_report = ComplianceReport(**latest_completed["result"])
        else:
            # No report available, return placeholder
            return {
                "current_score": 0,
                "status": "no_data",
                "checks": [],
                "summary": {
                    "total_checks": 0,
                    "compliant": 0,
                    "needs_review": 0,
                    "non_compliant": 0,
                    "total_gaps": 0,
                    "total_evidence": 0
                },
                "generated_at": None,
                "message": "No compliance data available. Generate a report first."
            }
    
    # Get historical trend (last 30 days of tasks)
    task_history = await task_manager.get_task_history(limit=50)
    historical_scores = []
    
    for task in task_history:
        if task.get("status") == "completed" and task.get("result"):
            result = task["result"]
            completed_at = datetime.fromisoformat(task["completed_at"])
            if (datetime.utcnow() - completed_at).days <= 30:
                historical_scores.append({
                    "date": completed_at.date().isoformat(),
                    "score": result["overall_score"],
                    "status": result["status"]
                })
    
    # Sort by date
    historical_scores.sort(key=lambda x: x["date"])
    
    dashboard_data = {
        "current_score": cached_report.overall_score,
        "status": cached_report.status.value,
        "checks": [
            {
                "category": check.category.value,
                "requirement": check.requirement,
                "status": check.status.value,
                "score": check.score,
                "gaps_count": len(check.gaps),
                "evidence_count": len(check.evidence),
                "metrics": check.metrics or {},
                "last_checked": check.last_checked.isoformat(),
                "next_check": check.next_check.isoformat()
            }
            for check in cached_report.checks
        ],
        "summary": cached_report.summary,
        "generated_at": cached_report.generated_at.isoformat(),
        "valid_until": cached_report.valid_until.isoformat(),
        "historical_trend": historical_scores,
        "last_updated": datetime.utcnow().isoformat()
    }
    
    await audit_service.log(
        action="compliance.dashboard_accessed",
        user_id=str(current_user.id),
        resource_type="compliance_dashboard",
        description="Accessed HIPAA compliance dashboard",
        is_phi_access=False,
        db=db
    )
    
    return dashboard_data


@router.get("/hipaa/breach-alerts")
async def get_breach_alerts(
    severity: Optional[str] = Query(None, regex="^(low|medium|high|critical)$"),
    active_only: bool = Query(True, description="Show only active alerts"),
    current_user: User = Depends(require_role(["super_admin", "admin"])),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends(get_audit_service)
):
    """Get breach alerts from the latest compliance check"""
    
    # Get latest report to extract breach alerts
    cached_report = await task_manager.get_cached_report(max_age_minutes=60)
    
    if not cached_report:
        return {"alerts": [], "message": "No recent compliance data available"}
    
    # Extract breach alerts from the breach notification check
    breach_alerts = []
    for check in cached_report.checks:
        if check.category.value == "breach_notification" and check.metrics:
            alerts = check.metrics.get("active_breach_alerts", 0)
            if alerts > 0:
                # This would normally come from the actual breach detection
                # For now, return placeholder data
                breach_alerts.append({
                    "alert_id": f"breach_{datetime.utcnow().timestamp()}",
                    "severity": "high",
                    "description": f"{alerts} breach alerts detected in latest scan",
                    "affected_users": 0,
                    "affected_records": 0,
                    "detected_at": check.last_checked.isoformat(),
                    "investigation_required": True,
                    "auto_notify_threshold": False
                })
    
    # Filter by severity if specified
    if severity:
        breach_alerts = [alert for alert in breach_alerts if alert["severity"] == severity]
    
    await audit_service.log(
        action="compliance.breach_alerts_accessed",
        user_id=str(current_user.id),
        resource_type="breach_alerts",
        description=f"Accessed breach alerts: {len(breach_alerts)} alerts found",
        is_phi_access=False,
        db=db
    )
    
    return {
        "alerts": breach_alerts,
        "total_count": len(breach_alerts),
        "last_checked": cached_report.generated_at.isoformat()
    }


@router.get("/hipaa/export")
async def export_compliance_report(
    format: str = Query("json", regex="^(json|csv|pdf)$"),
    task_id: Optional[str] = Query(None, description="Specific task ID to export"),
    current_user: User = Depends(require_role(["super_admin", "admin"])),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends(get_audit_service)
):
    """Export compliance report in various formats"""
    
    # Get report data
    if task_id:
        task_status = await task_manager.get_task_status(task_id)
        if task_status["status"] != "completed" or not task_status.get("result"):
            raise HTTPException(status_code=404, detail="Report not found or not completed")
        report_data = task_status["result"]
        report = ComplianceReport(**report_data)
    else:
        # Get latest cached report
        report = await task_manager.get_cached_report(max_age_minutes=60)
        if not report:
            raise HTTPException(status_code=404, detail="No recent report available")
    
    await audit_service.log(
        action="compliance.report_exported",
        user_id=str(current_user.id),
        resource_type="compliance_report",
        description=f"Exported compliance report in {format.upper()} format",
        request_data={"format": format, "task_id": task_id},
        is_phi_access=False,
        db=db
    )
    
    if format == "json":
        # Return JSON response
        return JSONResponse(
            content=report.__dict__,
            headers={
                "Content-Disposition": f"attachment; filename=hipaa_compliance_report_{datetime.utcnow().date()}.json"
            }
        )
    
    elif format == "csv":
        # Generate CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["Category", "Requirement", "Status", "Score", "Gaps", "Evidence", "Last Checked"])
        
        # Data rows
        for check in report.checks:
            writer.writerow([
                check.category.value,
                check.requirement,
                check.status.value,
                check.score,
                "; ".join(check.gaps),
                "; ".join(check.evidence),
                check.last_checked.isoformat()
            ])
        
        # Summary row
        writer.writerow([])
        writer.writerow(["Summary"])
        writer.writerow(["Overall Score", report.overall_score])
        writer.writerow(["Status", report.status.value])
        writer.writerow(["Total Checks", report.summary["total_checks"]])
        writer.writerow(["Compliant", report.summary["compliant"]])
        writer.writerow(["Needs Review", report.summary["needs_review"]])
        writer.writerow(["Non-Compliant", report.summary["non_compliant"]])
        writer.writerow(["Total Gaps", report.summary["total_gaps"]])
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=hipaa_compliance_report_{datetime.utcnow().date()}.csv"
            }
        )
    
    elif format == "pdf":
        # Generate PDF
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("HIPAA Compliance Report", title_style))
        story.append(Spacer(1, 12))
        
        # Report info
        story.append(Paragraph(f"<b>Generated:</b> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC", styles["Normal"]))
        story.append(Paragraph(f"<b>Overall Score:</b> {report.overall_score:.1f}/100", styles["Normal"]))
        story.append(Paragraph(f"<b>Status:</b> {report.status.value.upper()}", styles["Normal"]))
        story.append(Spacer(1, 20))
        
        # Summary table
        summary_data = [
            ["Metric", "Value"],
            ["Total Checks", str(report.summary["total_checks"])],
            ["Compliant", str(report.summary["compliant"])],
            ["Needs Review", str(report.summary["needs_review"])],
            ["Non-Compliant", str(report.summary["non_compliant"])],
            ["Total Gaps", str(report.summary["total_gaps"])]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Detailed checks
        story.append(Paragraph("Detailed Compliance Checks", styles["Heading2"]))
        story.append(Spacer(1, 12))
        
        for check in report.checks:
            story.append(Paragraph(f"<b>{check.category.value.upper()}</b>", styles["Heading3"]))
            story.append(Paragraph(f"<b>Requirement:</b> {check.requirement}", styles["Normal"]))
            story.append(Paragraph(f"<b>Status:</b> {check.status.value}", styles["Normal"]))
            story.append(Paragraph(f"<b>Score:</b> {check.score:.1f}/100", styles["Normal"]))
            
            if check.gaps:
                story.append(Paragraph("<b>Gaps:</b>", styles["Normal"]))
                for gap in check.gaps:
                    story.append(Paragraph(f"• {gap}", styles["Normal"]))
            
            if check.evidence:
                story.append(Paragraph("<b>Evidence:</b>", styles["Normal"]))
                for evidence in check.evidence:
                    story.append(Paragraph(f"• {evidence}", styles["Normal"]))
            
            story.append(Spacer(1, 12))
        
        doc.build(story)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=hipaa_compliance_report_{datetime.utcnow().date()}.pdf"
            }
        )


@router.get("/tasks/history")
async def get_task_history(
    limit: int = Query(50, le=100, description="Maximum number of tasks to return"),
    current_user: User = Depends(require_role(["super_admin", "admin"])),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends(get_audit_service)
):
    """Get history of compliance tasks"""
    
    tasks = await task_manager.get_task_history(limit=limit)
    
    await audit_service.log(
        action="compliance.task_history_accessed",
        user_id=str(current_user.id),
        resource_type="compliance_tasks",
        description=f"Accessed compliance task history: {len(tasks)} tasks",
        is_phi_access=False,
        db=db
    )
    
    return {
        "tasks": tasks,
        "total_count": len(tasks),
        "limit": limit
    }


@router.post("/hipaa/check-now")
async def trigger_immediate_check(
    priority: str = Query("high", regex="^(normal|high)$"),
    current_user: User = Depends(require_role(["super_admin", "admin"])),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends(get_audit_service)
):
    """Trigger immediate compliance check"""
    
    task_id = await task_manager.submit_compliance_report_task(
        user_id=str(current_user.id),
        priority=priority
    )
    
    await audit_service.log(
        action="compliance.immediate_check_triggered",
        user_id=str(current_user.id),
        resource_type="compliance_task",
        resource_id=task_id,
        description=f"Triggered immediate compliance check with priority {priority}",
        is_phi_access=False,
        db=db
    )
    
    return {
        "message": "Immediate compliance check started",
        "task_id": task_id,
        "priority": priority,
        "estimated_time": "2-5 minutes",
        "check_status_url": f"/api/v1/compliance/task/{task_id}"
    }
