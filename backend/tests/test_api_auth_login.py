import jwt
import os
import asyncio
import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta
from backend.app.models.user import User
from backend.app.security.password import password_service
from app.main import app

@pytest.mark.asyncio
async def test_demo_user_login_success(async_client: AsyncClient, demo_user):
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "username": "doctor",
            "is_demo": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["role"] == "doctor"
    assert data["user"]["is_demo"] == True

@pytest.mark.asyncio
async def test_real_user_login_success(async_client: AsyncClient, real_user):
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "username": "realuser",
            "password": "SecurePass123!",
            "is_demo": False
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_login_invalid_password(async_client: AsyncClient, real_user):
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "username": "realuser",
            "password": "wrongpass",
            "is_demo": False
        }
    )
    
    assert response.status_code == 401
    assert "Invalid username or password" in response.text

@pytest.mark.asyncio
async def test_login_user_not_found(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "username": "nouser",
            "password": "irrelevant",
            "is_demo": False
        }
    )
    
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_login_inactive_user(async_client: AsyncClient, inactive_user):
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "username": "inactive",
            "password": "password",
            "is_demo": False
        }
    )
    
    assert response.status_code == 401
    assert "disabled" in response.text.lower()

@pytest.mark.asyncio
async def test_demo_login_by_role(async_client: AsyncClient, demo_users):
    response = await async_client.post(
        "/api/v1/auth/demo-login",
        json={"role": "doctor"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["role"] == "doctor"

@pytest.mark.asyncio
async def test_login_rate_limiting(async_client: AsyncClient, demo_user):
    # Make 6 requests (limit is 5)
    for i in range(6):
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "username": "doctor",
                "is_demo": True
            }
        )
        
        if i < 5:
            assert response.status_code in [200, 401]
        else:
            assert response.status_code == 429  # Too Many Requests

@pytest.mark.asyncio
async def test_login_updates_last_login(async_client: AsyncClient, real_user, db_session):
    user = await db_session.get(User, real_user.id)
    old_last_login = user.last_login or datetime.utcnow() - timedelta(days=1)
    
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "username": "realuser",
            "password": "SecurePass123!",
            "is_demo": False
        }
    )
    
    assert response.status_code == 200
    await db_session.refresh(user)
    assert user.last_login is not None
    assert user.last_login > old_last_login

@pytest.mark.asyncio
async def test_demo_user_requires_no_password(async_client: AsyncClient, demo_user):
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "username": "doctor",
            "is_demo": True
        }
    )
    
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_real_user_requires_password(async_client: AsyncClient, real_user):
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "username": "realuser",
            "is_demo": False
        }
    )
    
    assert response.status_code == 400
    assert "password" in response.text.lower()

@pytest.mark.asyncio
async def test_demo_login_role_not_found(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/auth/demo-login",
        json={"role": "nonexistentrole"}
    )
    
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_login_returns_user_info(async_client: AsyncClient, real_user):
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "username": "realuser",
            "password": "SecurePass123!",
            "is_demo": False
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    user = data["user"]
    for field in ["id", "username", "email", "first_name", "last_name", "role", "is_demo"]:
        assert field in user

@pytest.mark.asyncio
async def test_login_token_can_access_protected_endpoint(async_client, real_user):
    # Login and get access token
    resp = await async_client.post("/api/v1/auth/login", json={"username": "realuser", "password": "SecurePass123!", "is_demo": False})
    assert resp.status_code == 200
    access_token = resp.json()["access_token"]
    # Access protected endpoint
    headers = {"Authorization": f"Bearer {access_token}"}
    resp2 = await async_client.get("/api/v1/auth/me", headers=headers)
    assert resp2.status_code == 200
    user = resp2.json()
    assert user["username"] == "realuser"

@pytest.mark.asyncio
async def test_refresh_token_works_after_login(async_client, real_user):
    resp = await async_client.post("/api/v1/auth/login", json={"username": "realuser", "password": "SecurePass123!", "is_demo": False})
    assert resp.status_code == 200
    refresh_token = resp.json()["refresh_token"]
    resp2 = await async_client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp2.status_code == 200
    data = resp2.json()
    assert "access_token" in data

@pytest.mark.asyncio
async def test_audit_log_created_on_login(async_client, real_user, db_session):
    # This test assumes audit logs are stored in a table or can be queried
    # Login
    resp = await async_client.post("/api/v1/auth/login", json={"username": "realuser", "password": "SecurePass123!", "is_demo": False})
    assert resp.status_code == 200
    # Wait for audit log to be written (if async)
    await asyncio.sleep(0.2)
    # Query audit log (example: AuditLog model)
    from backend.app.models.audit_log import AuditLog
    logs = (await db_session.execute(
        AuditLog.__table__.select().where(AuditLog.event_type == "LOGIN_SUCCESS")
    )).fetchall()
    assert any(str(real_user.id) in str(row.details) for row in logs)

@pytest.mark.asyncio
async def test_permissions_included_in_token(async_client, demo_user):
    resp = await async_client.post("/api/v1/auth/login", json={"username": "doctor", "is_demo": True})
    assert resp.status_code == 200
    access_token = resp.json()["access_token"]
    # Decode JWT without verification (for test only)
    payload = jwt.decode(access_token, options={"verify_signature": False})
    perms = payload.get("permissions", [])
    assert isinstance(perms, list)
    # Example: check for doctor permissions
    assert "patient:read" in perms or "ecg:analyze" in perms
