import uuid
import logging
from typing import Annotated, Optional
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from api.config import get_config, ServerConfig

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

async def verify_api_key(
    config: Annotated[ServerConfig, Depends(get_config)],
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]
) -> None:
    # If no key is configured, auth is disabled
    if not config.api_key:
        return
    
    # If a key is configured, but no credentials provided
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key"
        )
    
    # Validate key
    if credentials.credentials != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    
    logger.debug("Request authorized with API key.")

ApiKeyDependency = Depends(verify_api_key)
