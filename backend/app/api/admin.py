"""Admin API endpoints for user management and system settings."""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
from ..security.dependencies import get_current_user, require_permission
from ..security.permissions import Permission
from ..models.user import User

router = APIRouter(prefix="/api/admin", tags=["admin"])


class UserSettings(BaseModel):
    language: str
    theme: str
    notifications: bool
    timezone: str = "Europe/Istanbul"


class User(BaseModel):
    id: int
    username: str
    email: str
    role: str
    active: bool
    created_at: str


@router.get("/settings")
@require_permission(Permission.SETTINGS_READ)
async def get_user_settings(current_user: User = Depends(get_current_user)) -> UserSettings:
    """Get current user settings."""
    return UserSettings(
        language="tr",
        theme="light",
        notifications=True,
        timezone="Europe/Istanbul"
    )


@router.put("/settings")
@require_permission(Permission.SETTINGS_WRITE)
async def update_user_settings(settings: UserSettings, current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Update user settings."""
    # TODO: Save to database
    return {
        "success": True,
        "settings": settings.dict()
    }


@router.get("/users")
@require_permission(Permission.USERS_READ)
async def get_users(current_user: User = Depends(get_current_user)) -> List[User]:
    """Get list of all users."""
    # TODO: Fetch from database
    return [
        User(
            id=1,
            username="admin",
            email="admin@vitalstream.com",
            role="admin",
            active=True,
            created_at=datetime.now().isoformat()
        ),
        User(
            id=2,
            username="doctor1",
            email="doctor1@vitalstream.com",
            role="doctor",
            active=True,
            created_at=datetime.now().isoformat()
        ),
        User(
            id=3,
            username="nurse1",
            email="nurse1@vitalstream.com",
            role="nurse",
            active=True,
            created_at=datetime.now().isoformat()
        )
    ]


@router.get("/audit-logs")
@require_permission(Permission.AUDIT_READ)
async def get_audit_logs(current_user: User = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """Get audit logs."""
    # TODO: Fetch from database
    return [
        {
            "id": 1,
            "user_id": 1,
            "username": "admin",
            "action": "login",
            "resource": "auth",
            "ip_address": "192.168.1.100",
            "timestamp": datetime.now().isoformat(),
            "details": "Successful login"
        },
        {
            "id": 2,
            "user_id": 2,
            "username": "doctor1",
            "action": "view_patient",
            "resource": "patient:123",
            "ip_address": "192.168.1.101",
            "timestamp": datetime.now().isoformat(),
            "details": "Viewed patient record"
        }
    ]


@router.get("/system-health")
@require_permission(Permission.ANALYTICS_DASHBOARD)
async def get_system_health(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Get system health metrics."""
    return {
        "status": "healthy",
        "uptime": 86400,  # seconds
        "cpu_usage": 45.2,
        "memory_usage": 62.8,
        "disk_usage": 38.5,
        "active_connections": 142,
        "database": {
            "status": "healthy",
            "connections": 25,
            "response_time": 12.3
        },
        "cache": {
            "status": "healthy",
            "hit_rate": 87.5,
            "memory_usage": 45.2
        },
        "services": [
            {"name": "API", "status": "healthy", "response_time": 15.2},
            {"name": "WebSocket", "status": "healthy", "active_connections": 138},
            {"name": "Database", "status": "healthy", "response_time": 12.3},
            {"name": "Cache", "status": "healthy", "hit_rate": 87.5}
        ]
    }
