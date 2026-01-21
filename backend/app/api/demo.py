from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..core.cache import cache_service, CacheKeys
from ..services.demo_data_manager import (
    demo_data_manager, 
    DemoGenerationInProgressError, 
    DemoResetInProgressError
)
from ..models.demo_backup import DemoBackup, DemoResetHistory
from ..limiter import limiter
from ..security.rbac import rbac_service
from ..audit.audit_service import audit_service
from ..config import settings

logger = logging.getLogger(__name__)

class BackupRequest(BaseModel):
    name: Optional[str] = Field(None, description="Backup name")
    description: Optional[str] = Field(None, description="Backup description")
    backup_type: str = Field('selective', description="Backup type: full or selective")
    include_patients: bool = Field(True, description="Include patients in backup")
    include_ecg_data: bool = Field(True, description="Include ECG data in backup")
    include_alerts: bool = Field(True, description="Include alerts in backup")
    include_devices: bool = Field(False, description="Include devices in backup")

class BackupResponse(BaseModel):
    success: bool
    backup_id: Optional[str] = None
    name: str
    size_bytes: int
    created_at: str
    duration_ms: int
    error: Optional[str] = None

class RestoreRequest(BaseModel):
    backup_id: str = Field(..., description="Backup ID to restore from")
    confirm: bool = Field(..., description="Confirmation for restore operation")

class RestoreResponse(BaseModel):
    success: bool
    backup_id: str
    backup_name: str
    restored_at: str
    duration_ms: int
    records_restored: int
    error: Optional[str] = None

class ResetHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    total_count: int

# Existing models...
class ScenarioResponse(BaseModel):
    role: str
    title: str
    description: str
    patients: List[Dict[str, Any]]
    devices: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    settings: Dict[str, Any]
    generated_at: str
    cache_ttl: int

class ScenarioListResponse(BaseModel):
    scenarios: List[Dict[str, Any]]
    total_count: int
    cached_count: int

class ResetRequest(BaseModel):
    confirm: bool = Field(..., description="Confirmation for destructive operation")
    role: Optional[str] = Field(None, description="Specific role to reset (optional)")

class StatusResponse(BaseModel):
    status: str
    initialized_at: str
    version: str
    statistics: Dict[str, Any]
    last_reset: Optional[str] = None
    regeneration_in_progress: bool

class ResetRequest(BaseModel):
    confirm: bool = Field(..., description="Confirmation for destructive operation")
    role: Optional[str] = Field(None, description="Specific role to reset (optional)")
    create_backup: bool = Field(False, description="Create backup before reset")
    reset_patients: bool = Field(True, description="Reset patients data")
    reset_ecg_data: bool = Field(True, description="Reset ECG data")
    reset_alerts: bool = Field(True, description="Reset alerts data")
    reset_devices: bool = Field(False, description="Reset devices data")

class ResetResponse(BaseModel):
    success: bool
    message: str
    reset_id: str
    backup_id: Optional[str] = None
    reset_at: str
    duration_ms: int
    error: Optional[str] = None

class StatisticsResponse(BaseModel):
    patients: int
    ecg_sessions: int
    alarms: int
    calculated_at: str
    cache_ttl: int
    storage_usage_mb: float

class GenerateRequest(BaseModel):
    force: bool = Field(False, description="Force regeneration even if not needed")
    roles: Optional[List[str]] = Field(None, description="Specific roles to generate")

# Create router
router = APIRouter(
    prefix="/api/v1/demo",
    tags=["demo"],
    responses={
        423: {"description": "Demo operation in progress"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Service unavailable - maintenance mode"}
    }
)

@router.get(
    "/scenarios/{role}",
    response_model=ScenarioResponse,
    summary="Get role-specific demo scenario",
    description="Retrieve a complete demo scenario tailored for the specified healthcare role"
)
@limiter.limit("100/minute")
async def get_role_scenario(
    role: str,
    request: Request
) -> ScenarioResponse:
    """Load role-specific demo scenario"""
    
    # Validate role
    valid_roles = list(demo_data_manager.generator.SCENARIOS.keys())
    if role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Valid roles: {', '.join(valid_roles)}"
        )
    
    try:
        # Get user ID for audit
        user_id = "demo_user"  # Simplified for demo
        
        # Get scenario (with caching)
        scenario = await demo_data_manager.get_scenario(role, user_id)
        
        return ScenarioResponse(**scenario)
        
    except DemoGenerationInProgressError as e:
        raise HTTPException(
            status_code=423,
            detail=str(e),
            headers={"Retry-After": "120"}
        )
    except Exception as e:
        logger.error(f"Error getting scenario for role {role}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load scenario"
        )

@router.get(
    "/scenarios",
    response_model=ScenarioListResponse,
    summary="List all available demo scenarios",
    description="Get a list of all available demo scenarios with metadata"
)
@limiter.limit("50/minute")
async def list_scenarios(
    request: Request
) -> ScenarioListResponse:
    """List all available scenarios"""
    
    try:
        user_id = "demo_user"  # Simplified for demo
        scenarios = await demo_data_manager.get_all_scenarios(user_id)
        
        # Calculate statistics
        total_count = len(scenarios)
        cached_count = sum(1 for s in scenarios if s.get("cached", False))
        
        return ScenarioListResponse(
            scenarios=scenarios,
            total_count=total_count,
            cached_count=cached_count
        )
        
    except Exception as e:
        logger.error(f"Error listing scenarios: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list scenarios"
        )

@router.post(
    "/reset",
    summary="Reset all demo data",
    description="Completely reset all demo data and regenerate from scratch"
)
@limiter.limit("10/minute")
async def reset_demo_data(
    request: ResetRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Reset all demo data"""
    
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required for destructive operation"
        )
    
    try:
        user_id = "demo_user"  # Simplified for demo
        
        # Perform reset with optional backup
        result = await demo_data_manager.reset_all(
            user_id=user_id, 
            create_backup=request.create_backup
        )
        
        # Schedule background regeneration
        background_tasks.add_task(
            demo_data_manager._background_regenerate_all,
            user_id
        )
        
        return result
        
    except DemoResetInProgressError as e:
        raise HTTPException(
            status_code=423,
            detail=str(e),
            headers={"Retry-After": "120"}
        )
    except Exception as e:
        logger.error(f"Error resetting demo data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to reset demo data"
        )

@router.post(
    "/reset/{role}",
    summary="Reset specific role data",
    description="Reset demo data for a specific healthcare role"
)
@limiter.limit("20/minute")
async def reset_role_data(
    role: str,
    request: ResetRequest
) -> Dict[str, Any]:
    """Reset specific role data"""
    
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required for destructive operation"
        )
    
    # Validate role
    valid_roles = list(demo_data_manager.generator.SCENARIOS.keys())
    if role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Valid roles: {', '.join(valid_roles)}"
        )
    
    try:
        user_id = "demo_user"  # Simplified for demo
        result = await demo_data_manager.reset_role(role, user_id)
        return result
        
    except DemoResetInProgressError as e:
        raise HTTPException(
            status_code=423,
            detail=str(e),
            headers={"Retry-After": "120"}
        )
    except Exception as e:
        logger.error(f"Error resetting role {role} data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset role {role} data"
        )

@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Get demo system status",
    description="Get current status of the demo system including statistics"
)
@limiter.limit("200/minute")
async def get_demo_status(
    request: Request
) -> StatusResponse:
    """Get demo system status"""
    
    try:
        status = await demo_data_manager.get_status()
        
        # Check if any operations are in progress
        regeneration_lock = await cache_service.exists(
            CacheKeys.DEMO_LOCK.format(operation="regenerate")
        )
        reset_lock = await cache_service.exists(
            CacheKeys.DEMO_LOCK.format(operation="reset_all")
        )
        
        status["regeneration_in_progress"] = regeneration_lock or reset_lock
        
        return StatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Error getting demo status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get demo status"
        )

@router.post(
    "/generate",
    summary="Regenerate demo data",
    description="Trigger background regeneration of demo data"
)
@limiter.limit("5/minute")
async def regenerate_demo_data(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Regenerate demo data in background"""
    
    try:
        user_id = "demo_user"  # Simplified for demo
        
        # Start background regeneration
        result = await demo_data_manager.regenerate_data(
            background_tasks, user_id
        )
        
        return result
        
    except DemoGenerationInProgressError as e:
        raise HTTPException(
            status_code=423,
            detail=str(e),
            headers={"Retry-After": "120"}
        )
    except Exception as e:
        logger.error(f"Error regenerating demo data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to regenerate demo data"
        )

@router.get(
    "/statistics",
    response_model=StatisticsResponse,
    summary="Get demo data statistics",
    description="Get comprehensive statistics about demo data usage"
)
@limiter.limit("100/minute")
async def get_demo_statistics(
    request: Request,
    include_storage: bool = Query(False, description="Include storage usage calculation")
) -> StatisticsResponse:
    """Get demo data statistics"""
    
    try:
        stats = await demo_data_manager.get_statistics()
        
        # Add storage usage if requested
        if include_storage:
            # This would calculate actual storage usage
            # For now, estimate based on data volume
            stats["storage_usage_mb"] = 500.0  # Placeholder
        
        return StatisticsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting demo statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get demo statistics"
        )

# Health check endpoint
@router.get(
    "/health",
    summary="Demo service health check",
    description="Health check for demo service components"
)
async def health_check() -> Dict[str, Any]:
    """Health check for demo service"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # Check cache service
    try:
        await cache_service.set("health_check", "ok", ttl=10)
        cache_result = await cache_service.get("health_check")
        health_status["checks"]["cache"] = "healthy" if cache_result == "ok" else "unhealthy"
    except Exception as e:
        health_status["checks"]["cache"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check scenario generator
    try:
        scenarios = list(demo_data_manager.generator.SCENARIOS.keys())
        health_status["checks"]["scenarios"] = "healthy" if scenarios else "unhealthy"
    except Exception as e:
        health_status["checks"]["scenarios"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Overall status
    if any("unhealthy" in str(check) for check in health_status["checks"].values()):
        health_status["status"] = "unhealthy"
    elif any("degraded" in str(check) for check in health_status["checks"].values()):
        health_status["status"] = "degraded"
    
    return health_status
