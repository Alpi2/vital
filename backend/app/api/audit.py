from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import datetime

from ..database import get_session
from ..models.user import User
from ..security.dependencies import get_current_user, require_permission
from ..security.permissions import Permission
from ..services.audit_service import audit_service
from ..schemas.audit import AuditLogResponse

router = APIRouter()


@router.get("/audit-logs", response_model=List[AuditLogResponse])
@require_permission(Permission.AUDIT_READ)
async def get_audit_logs(
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    success: Optional[bool] = Query(None),
    is_phi_access: Optional[bool] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Get audit logs with filters.

    Requires AUDIT_READ permission (Admin, Super Admin).
    """
    logs = await audit_service.get_audit_logs(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        start_date=start_date,
        end_date=end_date,
        success=success,
        is_phi_access=is_phi_access,
        limit=limit,
        offset=offset,
        db=db,
    )

    return logs
