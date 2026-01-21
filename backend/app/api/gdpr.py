"""
GDPR Compliance API Endpoints
Implements Articles 15-22 with proper security and rate limiting
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
import uuid
import secrets
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.database import get_db
from app.services.consent_service import ConsentService, ConsentRequest, ConsentResponse
from app.services.data_subject_rights_service import (
    DataSubjectRightsService, AccessRequest, ErasureRequest, PortabilityRequest
)
from app.services.anonymization_service import AnonymizationService, AnonymizationConfig
from app.models.gdpr import ConsentType, LegalBasisType, DataSubjectRequestType
from app.security import get_current_user, verify_patient_access
from app.limiter import limiter

router = APIRouter(prefix="/api/v1/gdpr", tags=["gdpr"])
security = HTTPBearer()

# Rate limiting: 10 requests per hour for sensitive operations
@limiter.limit("10/hour")
async def rate_limit_sensitive_operations(request: Request):
    pass


# Pydantic Models for API

class ConsentGrantRequest(BaseModel):
    """Request model for granting consent"""
    consent_type: ConsentType
    consent_text: str = Field(..., min_length=10, description="Exact consent text shown to user")
    purpose: str = Field(..., min_length=5, description="Purpose of data processing")
    legal_basis: LegalBasisType
    expiry_date: Optional[datetime] = Field(None, description="Consent expiry date")
    parental_consent_required: bool = Field(False, description="Parental consent for minors")
    parent_id: Optional[uuid.UUID] = Field(None, description="Parent/guardian ID")
    parent_verification_method: Optional[str] = Field(None, description="Parent verification method")

    @validator('consent_type')
    def validate_consent_type(cls, v):
        if v not in ConsentType:
            raise ValueError("Invalid consent type")
        return v


class ConsentRevokeRequest(BaseModel):
    """Request model for revoking consent"""
    consent_id: uuid.UUID
    reason: Optional[str] = Field(None, description="Reason for revocation")


class AccessRequestModel(BaseModel):
    """Request model for data access"""
    format: str = Field("json", regex="^(json|csv|fhir)$", description="Export format")
    delivery_method: str = Field("email", regex="^(email|download|api)$", description="Delivery method")
    data_categories: Optional[List[str]] = Field(None, description="Specific data categories")
    step_up_token: Optional[str] = Field(None, description="Step-up authentication token")


class ErasureRequestModel(BaseModel):
    """Request model for data erasure"""
    reason: str = Field(..., min_length=10, description="Reason for erasure request")
    exceptions: Optional[List[str]] = Field(None, description="Legal exceptions to consider")
    step_up_token: Optional[str] = Field(None, description="Step-up authentication token")


class PortabilityRequestModel(BaseModel):
    """Request model for data portability"""
    format: str = Field("fhir", regex="^(fhir|json)$", description="Export format")
    transfer_destination: Optional[str] = Field(None, description="Direct transfer destination")
    step_up_token: Optional[str] = Field(None, description="Step-up authentication token")


class StepUpVerificationRequest(BaseModel):
    """Request model for step-up verification"""
    method: str = Field("email", regex="^(email|sms)$", description="Verification method")


class BulkConsentRequest(BaseModel):
    """Request model for bulk consent updates"""
    consents: List[Dict] = Field(..., min_items=1, description="List of consent updates")


# Consent Management Endpoints

@router.post("/consent", response_model=Dict[str, Any])
async def grant_consent(
    request: ConsentGrantRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    http_request: Request = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Grant consent for data processing
    GDPR Article 7: Consent must be freely given, specific, informed, and unambiguous
    """
    try:
        # Verify patient access
        patient_id = await verify_patient_access(current_user, db)
        
        # Get IP and user agent for audit trail
        ip_address = get_remote_address(http_request)
        user_agent = http_request.headers.get("user-agent")
        
        # Create consent request
        consent_request = ConsentRequest(
            patient_id=patient_id,
            consent_type=request.consent_type,
            consent_text=request.consent_text,
            purpose=request.purpose,
            legal_basis=request.legal_basis,
            ip_address=ip_address,
            user_agent=user_agent,
            expiry_date=request.expiry_date,
            parental_consent_required=request.parental_consent_required,
            parent_id=request.parent_id,
            parent_verification_method=request.parent_verification_method
        )
        
        # Process consent
        consent_service = ConsentService(db)
        result = await consent_service.grant_consent(consent_request)
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        
        # Log consent granted for audit
        background_tasks.add_task(
            log_consent_activity,
            patient_id,
            "consent_granted",
            request.consent_type.value,
            ip_address,
            user_agent
        )
        
        return {
            "success": True,
            "consent_id": str(result.consent_id),
            "message": result.message,
            "granted_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to grant consent: {str(e)}")


@router.delete("/consent/{consent_id}", response_model=Dict[str, Any])
async def revoke_consent(
    consent_id: uuid.UUID,
    request: ConsentRevokeRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    http_request: Request = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Revoke consent for data processing
    GDPR Article 7(3): Right to withdraw consent at any time, as easy as granting it
    """
    try:
        # Verify patient access
        patient_id = await verify_patient_access(current_user, db)
        
        # Get IP and user agent for audit trail
        ip_address = get_remote_address(http_request)
        user_agent = http_request.headers.get("user-agent")
        
        # Process consent revocation
        consent_service = ConsentService(db)
        result = await consent_service.revoke_consent(
            consent_id, patient_id, request.reason, ip_address, user_agent
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        
        # Log consent revoked for audit
        background_tasks.add_task(
            log_consent_activity,
            patient_id,
            "consent_revoked",
            str(consent_id),
            ip_address,
            user_agent,
            request.reason
        )
        
        return {
            "success": True,
            "message": result.message,
            "revoked_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to revoke consent: {str(e)}")


@router.get("/consent", response_model=Dict[str, Any])
async def list_consents(
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: dict = Depends(get_current_user)
):
    """
    List all patient consents
    """
    try:
        # Verify patient access
        patient_id = await verify_patient_access(current_user, db)
        
        # Get consent history
        consent_service = ConsentService(db)
        consents = await consent_service.get_consent_history(patient_id)
        
        return {
            "success": True,
            "consents": [
                {
                    "consent_id": str(consent.id),
                    "consent_type": consent.consent_type.value,
                    "granted": consent.granted,
                    "granted_at": consent.granted_at.isoformat() if consent.granted_at else None,
                    "revoked_at": consent.revoked_at.isoformat() if consent.revoked_at else None,
                    "version": consent.version,
                    "purpose": consent.purpose,
                    "expiry_date": consent.expiry_date.isoformat() if consent.expiry_date else None
                }
                for consent in consents
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list consents: {str(e)}")


@router.post("/consent/bulk", response_model=Dict[str, Any])
async def bulk_update_consents(
    request: BulkConsentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    http_request: Request = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Update multiple consents in a single transaction
    """
    try:
        # Verify patient access
        patient_id = await verify_patient_access(current_user, db)
        
        # Get IP and user agent
        ip_address = get_remote_address(http_request)
        user_agent = http_request.headers.get("user-agent")
        
        # Process bulk update
        consent_service = ConsentService(db)
        result = await consent_service.bulk_update_consents(
            patient_id, request.consents, ip_address, user_agent
        )
        
        # Log bulk update for audit
        background_tasks.add_task(
            log_consent_activity,
            patient_id,
            "bulk_consent_update",
            f"{len(request.consents)} consents",
            ip_address,
            user_agent
        )
        
        return {
            "success": result.success,
            "message": result.message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to bulk update consents: {str(e)}")


@router.get("/consent/{consent_id}/proof", response_model=Dict[str, Any])
async def export_consent_proof(
    consent_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: dict = Depends(get_current_user)
):
    """
    Export consent proof with digital signature
    GDPR Article 7: Must be able to demonstrate consent
    """
    try:
        # Verify patient access
        patient_id = await verify_patient_access(current_user, db)
        
        # Export consent proof
        consent_service = ConsentService(db)
        proof_data = await consent_service.export_consent_proof(patient_id, consent_id)
        
        return {
            "success": True,
            "proof_data": proof_data,
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export consent proof: {str(e)}")


# Data Subject Rights Endpoints

@router.post("/access-request", response_model=Dict[str, Any])
@limiter.limit("10/hour")
async def create_access_request(
    request: AccessRequestModel,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    http_request: Request = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Create data access request (Article 15)
    Requires step-up authentication for security
    """
    try:
        # Verify patient access
        patient_id = await verify_patient_access(current_user, db)
        
        # Create access request
        access_request = AccessRequest(
            patient_id=patient_id,
            format=request.format,
            delivery_method=request.delivery_method,
            data_categories=request.data_categories
        )
        
        # Process request
        rights_service = DataSubjectRightsService(db)
        result = await rights_service.create_access_request(access_request, request.step_up_token)
        
        if not result.get("success"):
            if result.get("requires_verification"):
                # Initiate step-up verification
                verification = await rights_service.initiate_step_up_verification(
                    patient_id, "email"
                )
                return {
                    "success": False,
                    "requires_verification": True,
                    "verification_id": verification.get("verification_id"),
                    "expires_at": verification.get("expires_at"),
                    "message": "Step-up authentication required"
                }
            else:
                raise HTTPException(status_code=400, detail=result.get("message"))
        
        # Schedule background processing
        background_tasks.add_task(
            process_access_request_background,
            uuid.UUID(result["request_id"]),
            db
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create access request: {str(e)}")


@router.post("/erasure-request", response_model=Dict[str, Any])
@limiter.limit("10/hour")
async def create_erasure_request(
    request: ErasureRequestModel,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    http_request: Request = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Create data erasure request (Article 17)
    With proper legal exception handling
    """
    try:
        # Verify patient access
        patient_id = await verify_patient_access(current_user, db)
        
        # Create erasure request
        erasure_request = ErasureRequest(
            patient_id=patient_id,
            reason=request.reason,
            exceptions=request.exceptions
        )
        
        # Process request
        rights_service = DataSubjectRightsService(db)
        result = await rights_service.create_erasure_request(erasure_request, request.step_up_token)
        
        if not result.get("success"):
            if result.get("requires_verification"):
                # Initiate step-up verification
                verification = await rights_service.initiate_step_up_verification(
                    patient_id, "email"
                )
                return {
                    "success": False,
                    "requires_verification": True,
                    "verification_id": verification.get("verification_id"),
                    "expires_at": verification.get("expires_at"),
                    "message": "Step-up authentication required"
                }
            else:
                raise HTTPException(status_code=400, detail=result.get("message"))
        
        # Schedule background processing
        background_tasks.add_task(
            process_erasure_request_background,
            uuid.UUID(result["request_id"]),
            db
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create erasure request: {str(e)}")


@router.post("/portability-request", response_model=Dict[str, Any])
@limiter.limit("10/hour")
async def create_portability_request(
    request: PortabilityRequestModel,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    http_request: Request = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Create data portability request (Article 20)
    FHIR R4 format for healthcare interoperability
    """
    try:
        # Verify patient access
        patient_id = await verify_patient_access(current_user, db)
        
        # Create portability request
        portability_request = PortabilityRequest(
            patient_id=patient_id,
            format=request.format,
            transfer_destination=request.transfer_destination
        )
        
        # Process request
        rights_service = DataSubjectRightsService(db)
        result = await rights_service.create_portability_request(portability_request, request.step_up_token)
        
        if not result.get("success"):
            if result.get("requires_verification"):
                # Initiate step-up verification
                verification = await rights_service.initiate_step_up_verification(
                    patient_id, "email"
                )
                return {
                    "success": False,
                    "requires_verification": True,
                    "verification_id": verification.get("verification_id"),
                    "expires_at": verification.get("expires_at"),
                    "message": "Step-up authentication required"
                }
            else:
                raise HTTPException(status_code=400, detail=result.get("message"))
        
        # Schedule background processing
        background_tasks.add_task(
            process_portability_request_background,
            uuid.UUID(result["request_id"]),
            db
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create portability request: {str(e)}")


@router.get("/requests/{request_id}", response_model=Dict[str, Any])
async def get_request_status(
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: dict = Depends(get_current_user)
):
    """
    Get status of data subject request
    """
    try:
        # Verify patient access
        patient_id = await verify_patient_access(current_user, db)
        
        # Get request status
        rights_service = DataSubjectRightsService(db)
        result = await rights_service.get_request_status(request_id, patient_id)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("message"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get request status: {str(e)}")


@router.get("/download/{request_id}")
async def download_data_export(
    request_id: uuid.UUID,
    token: str,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Download data export with secure token
    """
    try:
        # Verify download token (implementation depends on your token system)
        if not verify_download_token(request_id, token):
            raise HTTPException(status_code=401, detail="Invalid or expired download token")
        
        # Get file path
        file_path = get_export_file_path(request_id)
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Export file not found")
        
        return FileResponse(
            file_path,
            media_type='application/zip',
            filename=f"data_export_{request_id}.zip"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download export: {str(e)}")


# Step-up Authentication

@router.post("/verify", response_model=Dict[str, Any])
async def initiate_step_up_verification(
    request: StepUpVerificationRequest,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: dict = Depends(get_current_user)
):
    """
    Initiate step-up authentication for sensitive operations
    """
    try:
        # Verify patient access
        patient_id = await verify_patient_access(current_user, db)
        
        # Initiate verification
        rights_service = DataSubjectRightsService(db)
        result = await rights_service.initiate_step_up_verification(
            patient_id, request.method
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate verification: {str(e)}")


# Privacy Policy and Documentation

@router.get("/privacy-policy", response_model=Dict[str, Any])
async def get_privacy_policy():
    """
    Get current privacy policy
    """
    try:
        from app.compliance.gdpr import GDPRCompliance
        
        gdpr = GDPRCompliance()
        policy = gdpr.get_privacy_policy()
        
        return {
            "success": True,
            "policy": policy
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get privacy policy: {str(e)}")


@router.get("/dpia", response_model=Dict[str, Any])
async def get_dpia_document():
    """
    Get Data Protection Impact Assessment document
    """
    try:
        from app.compliance.gdpr import GDPRCompliance
        
        gdpr = GDPRCompliance()
        dpia = gdpr.conduct_dpia("Patient Health Monitoring and ECG Analysis")
        
        return {
            "success": True,
            "dpia": dpia
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get DPIA: {str(e)}")


@router.get("/ropa", response_model=Dict[str, Any])
async def get_ropa_document():
    """
    Get Record of Processing Activities
    """
    try:
        from app.compliance.gdpr import GDPRCompliance
        
        gdpr = GDPRCompliance()
        ropa = gdpr.generate_ropa()
        
        return {
            "success": True,
            "ropa": ropa
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ROPA: {str(e)}")


# Background Task Functions

async def process_access_request_background(request_id: uuid.UUID, db: AsyncSession):
    """Background task to process access request"""
    try:
        rights_service = DataSubjectRightsService(db)
        await rights_service.process_access_request(request_id)
    except Exception as e:
        # Log error and notify admin
        print(f"Failed to process access request {request_id}: {str(e)}")


async def process_erasure_request_background(request_id: uuid.UUID, db: AsyncSession):
    """Background task to process erasure request"""
    try:
        rights_service = DataSubjectRightsService(db)
        # This would need admin approval in practice
        await rights_service.process_erasure_request(request_id, None)
    except Exception as e:
        # Log error and notify admin
        print(f"Failed to process erasure request {request_id}: {str(e)}")


async def process_portability_request_background(request_id: uuid.UUID, db: AsyncSession):
    """Background task to process portability request"""
    try:
        rights_service = DataSubjectRightsService(db)
        await rights_service.process_access_request(request_id)  # Similar to access request
    except Exception as e:
        # Log error and notify admin
        print(f"Failed to process portability request {request_id}: {str(e)}")


async def log_consent_activity(patient_id: uuid.UUID, action: str, details: str,
                              ip_address: str, user_agent: str, reason: str = None):
    """Log consent activity for audit trail"""
    # Implementation depends on your audit logging system
    pass


def verify_download_token(request_id: uuid.UUID, token: str) -> bool:
    """Verify download token is valid"""
    # Implementation depends on your token system
    return True


def get_export_file_path(request_id: uuid.UUID) -> Optional[str]:
    """Get file path for export"""
    # Implementation depends on your storage system
    return None
