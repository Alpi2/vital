"""
Enhanced Report API Endpoints
Complete implementation with comprehensive report generation, retrieval, and management
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
import io
import base64

from ..services.enhanced_report_service import EnhancedReportService, ReportType, ReportRequest
from ..services.enhanced_pdf_generator import EnhancedPDFGenerator
from ..services.enhanced_digital_signature import EnhancedDigitalSignatureService
from ..services.enhanced_report_encryption import EnhancedReportEncryption
from ..dependencies import get_current_user, get_database_service
from ..models.user import User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/reports", tags=["Reports"])

# Pydantic models for request/response
class ECGReportRequest(BaseModel):
    """Request model for ECG report generation"""
    session_id: str = Field(..., description="ECG analysis session ID")
    patient_id: Optional[str] = Field(None, description="Patient ID")
    format: str = Field("pdf", description="Report format (pdf, html)")
    include_signature: bool = Field(True, description="Include digital signature")
    include_encryption: bool = Field(True, description="Include encryption")
    convert_to_pdfa: bool = Field(True, description="Convert to PDF/A format")
    template_name: str = Field("ecg_report.html", description="Template name")

class HRVReportRequest(BaseModel):
    """Request model for HRV report generation"""
    patient_id: str = Field(..., description="Patient ID")
    time_range: str = Field("30d", description="Time range (7d, 30d, 90d)")
    format: str = Field("pdf", description="Report format (pdf, html)")
    include_signature: bool = Field(True, description="Include digital signature")
    include_encryption: bool = Field(True, description="Include encryption")
    convert_to_pdfa: bool = Field(True, description="Convert to PDF/A format")
    template_name: str = Field("hrv_report.html", description="Template name")

class AnomalyReportRequest(BaseModel):
    """Request model for anomaly report generation"""
    patient_id: str = Field(..., description="Patient ID")
    anomaly_ids: List[str] = Field(..., description="List of anomaly IDs")
    format: str = Field("pdf", description="Report format (pdf, html)")
    include_signature: bool = Field(True, description="Include digital signature")
    include_encryption: bool = Field(True, description="Include encryption")
    convert_to_pdfa: bool = Field(True, description="Convert to PDF/A format")
    template_name: str = Field("anomaly_report.html", description="Template name")

class ComprehensiveReportRequest(BaseModel):
    """Request model for comprehensive report generation"""
    patient_id: str = Field(..., description="Patient ID")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    format: str = Field("pdf", description="Report format (pdf, html)")
    include_signature: bool = Field(True, description="Include digital signature")
    include_encryption: bool = Field(True, description="Include encryption")
    convert_to_pdfa: bool = Field(True, description="Convert to PDF/A format")
    template_name: str = Field("comprehensive_report.html", description="Template name")

class BatchReportRequest(BaseModel):
    """Request model for batch report generation"""
    reports: List[Dict[str, Any]] = Field(..., description="List of report requests")
    parallel: bool = Field(True, description="Generate reports in parallel")
    max_concurrent: int = Field(10, description="Maximum concurrent generations")
    user_id: Optional[str] = Field(None, description="User ID for audit")

class ReportResponse(BaseModel):
    """Response model for report generation"""
    success: bool = Field(True, description="Generation success status")
    report_id: str = Field(..., description="Unique report identifier")
    report_type: str = Field(..., description="Type of report generated")
    patient_id: str = Field(..., description="Patient ID")
    file_path: str = Field(..., description="File path of generated report")
    file_size: int = Field(..., description="File size in bytes")
    generation_time: str = Field(..., description="Generation timestamp")
    signature_info: Optional[Dict[str, Any]] = Field(None, description="Digital signature information")
    encryption_info: Optional[Dict[str, Any]] = Field(None, description="Encryption information")
    pdfa_info: Optional[Dict[str, Any]] = Field(None, description="PDF/A conversion information")

# Initialize services
report_service = EnhancedReportService()
pdf_generator = EnhancedPDFGenerator()
signature_service = EnhancedDigitalSignatureService()
encryption_service = EnhancedReportEncryption()

@router.post("/ecg", response_model=ReportResponse)
async def generate_ecg_report(
    request: ECGReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Generate ECG analysis report
    
    - **session_id**: ECG analysis session ID
    - **patient_id**: Patient identifier
    - **format**: Report format (pdf, html)
    - **include_signature**: Whether to include digital signature
    - **include_encryption**: Whether to include encryption
    - **convert_to_pdfa**: Whether to convert to PDF/A format
    """
    
    try:
        # Validate session ID
        if not request.session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        # Generate report
        result = await report_service.generate_ecg_report(
            session_id=request.session_id,
            format=request.format,
            patient_id=request.patient_id,
            user_id=current_user.id,
            include_signature=request.include_signature,
            include_encryption=request.include_encryption,
            convert_to_pdfa=request.convert_to_pdfa,
            template_name=request.template_name
        )
        
        # Create response
        response = ReportResponse(
            success=True,
            report_id=result['metadata'].report_id,
            report_type="ecg_analysis",
            patient_id=result['metadata'].patient_id,
            file_path=result['file_path'],
            file_size=result['metadata'].file_size,
            generation_time=result['metadata'].generation_time.isoformat(),
            signature_info=result['metadata'].signature_info,
            encryption_info=result['metadata'].encryption_info,
            pdfa_info=result['metadata'].pdfa_info
        )
        
        logger.info(f"ECG report generated: {response.report_id}")
        return response
        
    except Exception as e:
        logger.error(f"ECG report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ECG report generation failed: {str(e)}")

@router.post("/hrv", response_model=ReportResponse)
async def generate_hrv_report(
    request: HRVReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Generate HRV analysis report
    
    - **patient_id**: Patient identifier
    - **time_range**: Time range for analysis (7d, 30d, 90d)
    - **format**: Report format (pdf, html)
    - **include_signature**: Whether to include digital signature
    - **include_encryption**: Whether to include encryption
    - **convert_to_pdfa**: Whether to convert to PDF/A format
    """
    
    try:
        # Validate patient ID
        if not request.patient_id:
            raise HTTPException(status_code=400, detail="Patient ID is required")
        
        # Validate time range
        valid_ranges = ["7d", "30d", "90d"]
        if request.time_range not in valid_ranges:
            raise HTTPException(status_code=400, detail=f"Invalid time range. Use: {', '.join(valid_ranges)}")
        
        # Generate report
        result = await report_service.generate_hrv_report(
            patient_id=request.patient_id,
            time_range=request.time_range,
            format=request.format,
            user_id=current_user.id,
            include_signature=request.include_signature,
            include_encryption=request.include_encryption,
            convert_to_pdfa=request.convert_to_pdfa,
            template_name=request.template_name
        )
        
        # Create response
        response = ReportResponse(
            success=True,
            report_id=result['metadata'].report_id,
            report_type="hrv_analysis",
            patient_id=result['metadata'].patient_id,
            file_path=result['file_path'],
            file_size=result['metadata'].file_size,
            generation_time=result['metadata'].generation_time.isoformat(),
            signature_info=result['metadata'].signature_info,
            encryption_info=result['metadata'].encryption_info,
            pdfa_info=result['metadata'].pdfa_info
        )
        
        logger.info(f"HRV report generated: {response.report_id}")
        return response
        
    except Exception as e:
        logger.error(f"HRV report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"HRV report generation failed: {str(e)}")

@router.post("/anomaly", response_model=ReportResponse)
async def generate_anomaly_report(
    request: AnomalyReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Generate anomaly detection report
    
    - **patient_id**: Patient identifier
    - **anomaly_ids**: List of anomaly IDs to include
    - **format**: Report format (pdf, html)
    - **include_signature**: Whether to include digital signature
    - **include_encryption**: Whether to include encryption
    - **convert_to_pdfa**: Whether to convert to PDF/A format
    """
    
    try:
        # Validate inputs
        if not request.patient_id:
            raise HTTPException(status_code=400, detail="Patient ID is required")
        
        if not request.anomaly_ids:
            raise HTTPException(status_code=400, detail="At least one anomaly ID is required")
        
        # Generate report
        result = await report_service.generate_anomaly_report(
            patient_id=request.patient_id,
            anomaly_ids=request.anomaly_ids,
            format=request.format,
            user_id=current_user.id,
            include_signature=request.include_signature,
            include_encryption=request.include_encryption,
            convert_to_pdfa=request.convert_to_pdfa,
            template_name=request.template_name
        )
        
        # Create response
        response = ReportResponse(
            success=True,
            report_id=result['metadata'].report_id,
            report_type="anomaly_report",
            patient_id=result['metadata'].patient_id,
            file_path=result['file_path'],
            file_size=result['metadata'].file_size,
            generation_time=result['metadata'].generation_time.isoformat(),
            signature_info=result['metadata'].signature_info,
            encryption_info=result['metadata'].encryption_info,
            pdfa_info=result['metadata'].pdfa_info
        )
        
        logger.info(f"Anomaly report generated: {response.report_id}")
        return response
        
    except Exception as e:
        logger.error(f"Anomaly report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly report generation failed: {str(e)}")

@router.post("/comprehensive", response_model=ReportResponse)
async def generate_comprehensive_report(
    request: ComprehensiveReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Generate comprehensive medical report
    
    - **patient_id**: Patient identifier
    - **start_date**: Start date for analysis (YYYY-MM-DD)
    - **end_date**: End date for analysis (YYYY-MM-DD)
    - **format**: Report format (pdf, html)
    - **include_signature**: Whether to include digital signature
    - **include_encryption**: Whether to include encryption
    - **convert_to_pdfa**: Whether to convert to PDF/A format
    """
    
    try:
        # Validate inputs
        if not request.patient_id:
            raise HTTPException(status_code=400, detail="Patient ID is required")
        
        # Parse dates
        try:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Check date range (limit to 1 year)
        if (end_date - start_date).days > 365:
            raise HTTPException(status_code=400, detail="Date range cannot exceed 1 year")
        
        # Generate report
        result = await report_service.generate_comprehensive_report(
            patient_id=request.patient_id,
            date_range=(start_date, end_date),
            format=request.format,
            user_id=current_user.id,
            include_signature=request.include_signature,
            include_encryption=request.include_encryption,
            convert_to_pdfa=request.convert_to_pdfa,
            template_name=request.template_name
        )
        
        # Create response
        response = ReportResponse(
            success=True,
            report_id=result['metadata'].report_id,
            report_type="comprehensive",
            patient_id=result['metadata'].patient_id,
            file_path=result['file_path'],
            file_size=result['metadata'].file_size,
            generation_time=result['metadata'].generation_time.isoformat(),
            signature_info=result['metadata'].signature_info,
            encryption_info=result['metadata'].encryption_info,
            pdfa_info=result['metadata'].pdfa_info
        )
        
        logger.info(f"Comprehensive report generated: {response.report_id}")
        return response
        
    except Exception as e:
        logger.error(f"Comprehensive report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comprehensive report generation failed: {str(e)}")

@router.post("/batch")
async def generate_batch_reports(
    request: BatchReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Generate multiple reports in batch
    
    - **reports**: List of report requests
    - **parallel**: Generate reports in parallel
    - **max_concurrent**: Maximum concurrent generations
    - **user_id**: User ID for audit logging
    """
    
    try:
        # Validate batch size
        if len(request.reports) > 100:
            raise HTTPException(status_code=400, detail="Batch size too large (maximum 100 reports)")
        
        # Convert to ReportRequest objects
        report_requests = []
        for report_data in request.reports:
            try:
                report_type = ReportType(report_data['report_type'])
                report_request = ReportRequest(
                    report_type=report_type,
                    patient_id=report_data['patient_id'],
                    session_id=report_data.get('session_id'),
                    time_range=report_data.get('time_range'),
                    anomaly_ids=report_data.get('anomaly_ids'),
                    format=report_data.get('format', 'pdf'),
                    include_signature=report_data.get('include_signature', True),
                    include_encryption=report_data.get('include_encryption', True),
                    convert_to_pdfa=report_data.get('convert_to_pdfa', True),
                    metadata=report_data.get('metadata'),
                    user_id=request.user_id or current_user.id,
                    access_level=report_data.get('access_level', 'standard')
                )
                report_requests.append(report_request)
            except Exception as e:
                logger.error(f"Invalid report request: {e}")
                continue
        
        # Generate batch reports
        results = await report_service.generate_batch_reports(
            requests=report_requests,
            user_id=request.user_id or current_user.id,
            parallel=request.parallel,
            max_concurrent=request.max_concurrent
        )
        
        # Create response
        batch_results = []
        for i, result in enumerate(results):
            if result.get('success'):
                batch_results.append({
                    'success': True,
                    'report_id': result['metadata'].report_id,
                    'report_type': result['metadata'].report_type,
                    'patient_id': result['metadata'].patient_id,
                    'file_path': result['file_path'],
                    'file_size': result['metadata'].file_size,
                    'generation_time': result['metadata'].generation_time.isoformat()
                })
            else:
                batch_results.append({
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'report_data': request.reports[i]
                })
        
        logger.info(f"Batch report generation completed: {len(batch_results)} reports")
        return {
            'success': True,
            'batch_size': len(request.reports),
            'successful_reports': len([r for r in batch_results if r['success']]),
            'failed_reports': len([r for r in batch_results if not r['success']]),
            'results': batch_results
        }
        
    except Exception as e:
        logger.error(f"Batch report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch report generation failed: {str(e)}")

@router.get("/{report_id}")
async def get_report(
    report_id: str,
    download: bool = Query(False, description="Download as file"),
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Retrieve a generated report
    
    - **report_id**: Unique report identifier
    - **download**: Whether to download as file
    """
    
    try:
        # Retrieve report
        result = report_service.retrieve_report(report_id)
        
        if download:
            # Return file for download
            file_path = Path(result['file_path'])
            return FileResponse(
                path=file_path,
                filename=f"{result['metadata'].report_type}_{result['metadata'].patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                media_type='application/pdf'
            )
        else:
            # Return report info
            return {
                'success': True,
                'metadata': result['metadata'],
                'file_path': result['file_path'],
                'file_size': len(result['report_data'])
            }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report not found")
    except Exception as e:
        logger.error(f"Report retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report retrieval failed: {str(e)}")

@router.get("/{report_id}/verify")
async def verify_report_signature(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Verify digital signature of a report
    
    - **report_id**: Unique report identifier
    """
    
    try:
        # Retrieve report
        result = report_service.retrieve_report(report_id)
        
        # Check if report has signature
        if not result['metadata'].signature_info:
            return {
                'success': False,
                'error': 'Report does not have a digital signature'
            }
        
        # Verify signature
        verification_result = signature_service.verify_signature(
            report_data=result['report_data'],
            signature_metadata=result['metadata'].signature_info,
            verify_chain=True
        )
        
        return {
            'success': True,
            'report_id': report_id,
            'verification_result': verification_result
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report not found")
    except Exception as e:
        logger.error(f"Signature verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Signature verification failed: {str(e)}")

@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    reason: str = Query(..., description="Reason for deletion"),
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Delete a report (admin only)
    
    - **report_id**: Unique report identifier
    - **reason**: Reason for deletion
    """
    
    try:
        # Check admin permissions
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        # Delete report
        success = report_service.delete_report(report_id, current_user.id, reason)
        
        if success:
            return {
                'success': True,
                'message': f'Report {report_id} deleted successfully'
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete report")
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report not found")
    except Exception as e:
        logger.error(f"Report deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report deletion failed: {str(e)}")

@router.get("/")
async def list_reports(
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    report_type: Optional[str] = Query(None, description="Filter by report type"),
    start_date: Optional[str] = Query(None, description="Filter by start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter by end date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of reports"),
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    List reports with filtering options
    
    - **patient_id**: Filter by patient ID
    - **report_type**: Filter by report type
    - **start_date**: Filter by start date (YYYY-MM-DD)
    - **end_date**: Filter by end date (YYYY-MM-DD)
    - **limit**: Maximum number of reports to return
    """
    
    try:
        # Parse date range
        date_range = None
        if start_date and end_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                date_range = (start_dt, end_dt)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # List reports
        reports = report_service.list_reports(
            patient_id=patient_id,
            report_type=report_type,
            date_range=date_range,
            limit=limit
        )
        
        return {
            'success': True,
            'total_reports': len(reports),
            'reports': reports
        }
        
    except Exception as e:
        logger.error(f"Report listing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report listing failed: {str(e)}")

@router.get("/statistics")
async def get_report_statistics(
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Get report generation statistics
    
    Requires admin privileges
    """
    
    try:
        # Check admin permissions
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        # Get statistics
        stats = report_service.get_report_statistics()
        
        return {
            'success': True,
            'statistics': stats
        }
        
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")

@router.post("/cleanup")
async def cleanup_old_reports(
    retention_days: int = Query(2555, ge=30, le=3650, description="Retention period in days"),
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Clean up old reports beyond retention period
    
    Requires admin privileges
    """
    
    try:
        # Check admin permissions
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        # Clean up old reports
        deleted_count = report_service.cleanup_old_reports(retention_days)
        
        return {
            'success': True,
            'deleted_reports': deleted_count,
            'retention_days': retention_days
        }
        
    except Exception as e:
        logger.error(f"Report cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report cleanup failed: {str(e)}")

@router.get("/templates")
async def list_report_templates(
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    List available report templates
    """
    
    try:
        templates = [
            {
                'name': 'ecg_report.html',
                'description': 'ECG Analysis Report',
                'report_type': 'ecg_analysis',
                'supported_formats': ['pdf', 'html']
            },
            {
                'name': 'hrv_report.html',
                'description': 'Heart Rate Variability Analysis Report',
                'report_type': 'hrv_analysis',
                'supported_formats': ['pdf', 'html']
            },
            {
                'name': 'anomaly_report.html',
                'description': 'Anomaly Detection Report',
                'report_type': 'anomaly_report',
                'supported_formats': ['pdf', 'html']
            },
            {
                'name': 'comprehensive_report.html',
                'description': 'Comprehensive Medical Report',
                'report_type': 'comprehensive',
                'supported_formats': ['pdf', 'html']
            }
        ]
        
        return {
            'success': True,
            'templates': templates
        }
        
    except Exception as e:
        logger.error(f"Template listing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Template listing failed: {str(e)}")

@router.get("/encryption/info")
async def get_encryption_info(
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Get encryption service information
    
    Requires admin privileges
    """
    
    try:
        # Check admin permissions
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        # Get encryption info
        info = encryption_service.get_encryption_info()
        
        return {
            'success': True,
            'encryption_info': info
        }
        
    except Exception as e:
        logger.error(f"Encryption info retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Encryption info retrieval failed: {str(e)}")

@router.post("/encryption/rotate-key")
async def rotate_encryption_key(
    reason: str = Query(..., description="Reason for key rotation"),
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Rotate encryption key
    
    Requires admin privileges
    """
    
    try:
        # Check admin permissions
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        # Rotate key
        success = encryption_service.rotate_key(current_user.id, reason)
        
        if success:
            return {
                'success': True,
                'message': 'Encryption key rotated successfully',
                'reason': reason
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to rotate encryption key")
        
    except Exception as e:
        logger.error(f"Key rotation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Key rotation failed: {str(e)}")

@router.get("/signature/info")
async def get_signature_info(
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Get digital signature service information
    
    Requires admin privileges
    """
    
    try:
        # Check admin permissions
        if current_user.role != 'admin':
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        # Get signature info
        info = signature_service.get_certificate_summary()
        
        return {
            'success': True,
            'signature_info': info
        }
        
    except Exception as e:
        logger.error(f"Signature info retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Signature info retrieval failed: {str(e)}")
