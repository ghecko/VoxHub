from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/health", tags=["System"])
async def health_check():
    return JSONResponse(content={"status": "healthy"})

@router.get("/v1/health", tags=["System"])
async def v1_health_check():
    return JSONResponse(content={"status": "healthy"})
