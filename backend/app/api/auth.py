
# -------------------- IMPORTS --------------------
from fastapi import APIRouter, HTTPException, Depends, status, Security, Request
from fastapi.security import HTTPAuthorizationCredentials
from datetime import timedelta, datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from ..config import settings
from ..database import get_session
from ..models.user import User
from ..models.role import Role
from ..schemas.auth import LoginRequest, DemoLoginRequest, TokenResponse, UserInfo
from ..security.jwt import jwt_service
from ..security.audit import audit_service
from ..security.audit import AuditLogger, AuditEventType, AuditSeverity
from ..security import (
    verify_token,
    blacklist_token,
    security,
    get_current_user,
)
from ..security.password import password_service
from ..limiter import limiter
from pydantic import BaseModel

# -------------------- ROUTER --------------------
router = APIRouter()

# Instantiate audit logger
audit_logger = AuditLogger()


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str


@router.post("/token")
async def token():
    """Return a development bearer token when `debug` is enabled.

    The endpoint returns a minimal JSON object with `access_token` and
    `token_type`. In production this endpoint returns 401 Unauthorized.
    """
    if settings.debug:
        return {"access_token": "dev-token", "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Not allowed in production")


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, login_data: LoginRequest, db: AsyncSession = Depends(get_session)):
    """
    User login endpoint supporting both demo and real users.

    - **Demo user login:** Set `is_demo=true`, no password required. Used for quick access to demo accounts by username.
    - **Real user login:** Set `is_demo=false`, password required. Used for authenticating real users.
    - **Rate limit:** 5 requests per minute per IP.

    **Request Body Examples:**

    Demo user:
    ```json
    {
        "username": "doctor",
        "is_demo": true
    }
    ```

    Real user:
    ```json
    {
        "username": "realuser",
        "password": "SecurePass123!",
        "is_demo": false
    }
    ```

    **Response Example:**
    ```json
    {
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "token_type": "bearer",
        "expires_in": 900,
        "user": {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "doctor",
            "email": "doctor@vitalstream.demo",
            "first_name": "Sarah",
            "last_name": "Johnson",
            "role": "doctor",
            "is_demo": true
        }
    }
    ```

    **Error Responses:**
    - 400: Missing required fields (e.g., password for real user)
    - 401: Invalid credentials or inactive user
    - 429: Too many requests (rate limit exceeded)
    - 500: Internal server error
    """
    try:
        # Query user by username
        result = await db.execute(select(User).where(User.username == login_data.username))
        user = result.scalar_one_or_none()
        ip = request.client.host if request.client and request.client.host else ""
        user_agent = request.headers.get("user-agent")

        if not user:
            audit_logger.log_authentication(
                event_type=AuditEventType.LOGIN_FAILURE,
                user_email=login_data.username,
                ip_address=ip,
                result="failure",
                details={"reason": "User not found", "user_agent": user_agent}
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

        if not user.is_active:
            audit_logger.log_authentication(
                event_type=AuditEventType.LOGIN_FAILURE,
                user_email=user.email,
                ip_address=ip,
                result="failure",
                details={"reason": "User inactive", "user_agent": user_agent}
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is disabled")

        if login_data.is_demo:
            if not user.is_demo:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User is not a demo user")
            # No password check for demo users
        else:
            if not login_data.password:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password is required for non-demo users")
            if not password_service.verify_password(login_data.password, user.password_hash):
                audit_logger.log_authentication(
                    event_type=AuditEventType.LOGIN_FAILURE,
                    user_email=user.email,
                    ip_address=ip,
                    result="failure",
                    details={"reason": "Invalid password", "user_agent": user_agent}
                )
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

        # Get role permissions
        role_result = await db.execute(select(Role).where(Role.name == user.role))
        role_obj = role_result.scalar_one_or_none()
        permissions = []
        if role_obj and getattr(role_obj, "permissions", None):
            perms = getattr(role_obj, "permissions", None)
            if isinstance(perms, dict):
                permissions = perms.get("permissions", [])

        # Generate tokens
        access_token = await jwt_service.create_access_token(
            user_id=str(user.id),
            email=user.email,
            username=user.username,
            role=user.role,
            permissions=permissions
        )
        refresh_token = await jwt_service.create_refresh_token(
            user_id=str(user.id),
            email=user.email
        )

        # Update last_login
        await db.execute(update(User).where(User.id == user.id).values(last_login=datetime.utcnow()))
        await db.commit()

        # Log successful login
        audit_logger.log_authentication(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_email=user.email,
            ip_address=ip,
            result="success",
            details={"user_id": str(user.id), "user_agent": user_agent}
        )


        # Calculate expires_in from settings or JWT service config
        expires_in = None
        if hasattr(settings, "access_token_expire_minutes"):
            expires_in = int(getattr(settings, "access_token_expire_minutes", 15)) * 60
        elif hasattr(jwt_service, "access_exp"):
            expires_in = int(getattr(jwt_service, "access_exp").total_seconds())
        else:
            expires_in = 900

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=expires_in,
            user=UserInfo(
                id=str(user.id),
                username=user.username,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                role=user.role,
                is_demo=user.is_demo
            ).dict()
        )
    except HTTPException:
        raise
    except Exception as e:
        ip = request.client.host if request.client and request.client.host else ""
        audit_logger.log_authentication(
            event_type=AuditEventType.LOGIN_FAILURE,
            user_email=login_data.username,
            ip_address=ip,
            result="failure",
            details={"reason": f"System error: {str(e)}"}
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred during login")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest, db: AsyncSession = Depends(get_session)):
    """Refresh access token using refresh token.
    
    This endpoint allows clients to obtain a new access token
    without requiring the user to log in again.
    """
    try:
        # Verify refresh token
        payload = await verify_token(request.refresh_token, "refresh")
        
        # Lookup user
        user_id = payload.get("user_id") or payload.get("sub")
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token: user not found")

        # Create new tokens
        access_token = await jwt_service.create_access_token(
            user_id=str(user.id),
            email=user.email,
            role=user.role
        )
        new_refresh_token = await jwt_service.create_refresh_token(
            user_id=str(user.id),
            email=user.email
        )
        
        # Blacklist old refresh token (token rotation)
        exp = datetime.fromtimestamp(payload["exp"])
        await blacklist_token(request.refresh_token, exp)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=jwt_service.access_token_expire_minutes * 60,
            user=UserInfo(
                id=str(user.id),
                username=user.username,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                role=user.role,
                is_demo=user.is_demo
            ).dict()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )

@router.post("/demo-login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def demo_login(request: Request, demo_data: DemoLoginRequest, db: AsyncSession = Depends(get_session)):
    """
    Quick demo user login by role. No password required.

    - **Description:** Instantly logs in as a demo user for the specified role (e.g., doctor, nurse, technician).
    - **Rate limit:** 10 requests per minute per IP.

    **Request Body Example:**
    ```json
    {
      "role": "doctor"
    }
    ```

    **Response Example:**
    ```json
    {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "token_type": "bearer",
      "expires_in": 900,
      "user": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "username": "doctor",
        "email": "doctor@vitalstream.demo",
        "first_name": "Sarah",
        "last_name": "Johnson",
        "role": "doctor",
        "is_demo": true
      }
    }
    ```

    **Error Responses:**
    - 404: No demo user found for the specified role
    - 429: Too many requests (rate limit exceeded)
    """
    # Query demo user by role
    result = await db.execute(
        select(User).where(
            User.role == demo_data.role,
            User.is_demo == True,
            User.is_active == True
        )
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No demo user found for role: {demo_data.role}"
        )
    # Use regular login logic
    return await login(
        request=request,
        login_data=LoginRequest(
            username=user.username,
            is_demo=True
        ),
        db=db
    )


async def get_role_permissions(role: str, db: AsyncSession) -> list[str]:
    """Get permissions for a role."""
    result = await db.execute(
        select(Role).where(Role.name == role)
    )
    role_obj = result.scalar_one_or_none()
    
    if not role_obj:
        return []

    # Safely extract permissions from the role object (could be a dict-like JSONB)
    perms = getattr(role_obj, "permissions", None)
    if not perms or not isinstance(perms, dict):
        return []

    return perms.get("permissions", [])


@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Logout user by blacklisting their token.
    
    This adds the current access token to a blacklist,
    preventing it from being used for future requests.
    """
    token = credentials.credentials
    
    try:
        # Verify token and get expiration
        payload = await verify_token(token, "access")
        from datetime import datetime
        exp = datetime.fromtimestamp(payload["exp"])
        
        # Blacklist token
        await blacklist_token(token, exp)
        
        return {"message": "Successfully logged out"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


@router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user information.
    
    This endpoint returns information about the currently
    authenticated user based on their access token.
    """
    return {
        "user_id": current_user.get("user_id"),
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "sub": current_user.get("sub"),
    }
