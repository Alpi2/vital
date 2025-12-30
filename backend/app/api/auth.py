from fastapi import APIRouter, HTTPException
from ..config import settings

router = APIRouter()


@router.post("/token")
async def token():
    """Development-only token endpoint.

    Returns a simple dev token when the app is running in debug mode. In
    production this endpoint will return 401 Unauthorized.
    """
    if settings.debug:
        return {"access_token": "dev-token", "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Not allowed in production")
