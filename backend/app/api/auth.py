from fastapi import APIRouter, HTTPException, Depends, status, Security
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from datetime import timedelta

from ..config import settings
from ..security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    blacklist_token,
    security,
    get_current_user,
)

"""
Authentication API

Provides endpoints for user authentication, token management, and logout.
Includes both development helpers and production-ready authentication.
"""

router = APIRouter()


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str


class LoginRequest(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str


@router.post("/token")
async def token():
    """Return a development bearer token when `debug` is enabled.

    The endpoint returns a minimal JSON object with `access_token` and
    `token_type`. In production this endpoint returns 401 Unauthorized.
    """
    if settings.debug:
        return {"access_token": "dev-token", "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Not allowed in production")


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="""Authenticate user and return JWT tokens.
    
**Request Body:**
- `email` (required): User's email address
- `password` (required): User's password

**Returns:**
- `access_token`: Short-lived token for API requests (default: 30 minutes)
- `refresh_token`: Long-lived token for refreshing access token (default: 7 days)
- `token_type`: Always "bearer"

**Authentication Flow:**
1. User submits email and password
2. Server validates credentials
3. Server generates access + refresh tokens
4. Client stores tokens securely
5. Client includes access token in Authorization header for API requests

**Example Request:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Example Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Note:** In debug mode, accepts any email/password combination.
    """,
    responses={
        200: {
            "description": "Login successful",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "token_type": "bearer"
                    }
                }
            }
        },
        401: {
            "description": "Invalid credentials",
            "content": {
                "application/json": {
                    "example": {"detail": "Incorrect email or password"}
                }
            }
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "email"],
                                "msg": "value is not a valid email address",
                                "type": "value_error.email"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def login(request: LoginRequest):
    """Authenticate user and return JWT tokens."""
    # TODO: Implement real authentication logic
    # - Query user from database
    # - Verify password hash
    # - Check user status
    
    if settings.debug:
        # Demo mode: accept any credentials
        user_data = {
            "sub": request.email,
            "user_id": 1,
            "email": request.email,
        }
    else:
        # Production: implement real authentication
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication not implemented. Please implement user authentication."
        )
    
    # Create tokens
    access_token = create_access_token(
        user_data,
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )
    refresh_token = create_refresh_token(user_data)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token.
    
    This endpoint allows clients to obtain a new access token
    without requiring the user to log in again.
    """
    try:
        # Verify refresh token
        payload = await verify_token(request.refresh_token, "refresh")
        
        # Extract user data
        user_data = {
            "sub": payload.get("sub"),
            "user_id": payload.get("user_id"),
            "email": payload.get("email"),
        }
        
        # Create new tokens
        access_token = create_access_token(
            user_data,
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        new_refresh_token = create_refresh_token(user_data)
        
        # Blacklist old refresh token (token rotation)
        from datetime import datetime
        exp = datetime.fromtimestamp(payload["exp"])
        await blacklist_token(request.refresh_token, exp)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )


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
        "email": current_user.get("email"),
        "sub": current_user.get("sub"),
    }
