from fastapi import APIRouter

router = APIRouter()


@router.post("/token")
async def token():
    return {"access_token": "dev-token", "token_type": "bearer"}
