from typing import Annotated, List
from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse
from api.config import get_config, ServerConfig
from api.transcriber import get_transcription_service, TranscriptionService
from core.registry import list_supported_models

router = APIRouter()

@router.get("/v1/models", tags=["Models"])
async def list_models():
    """
    List available models from models.yaml in OpenAI format.
    """
    supported = list_supported_models()
    models_data = []
    for model_id in supported:
        models_data.append({
            "id": model_id,
            "object": "model",
            "owned_by": "voxbench",
            "permission": []
        })
    return JSONResponse(content={"data": models_data})

@router.get("/models/list", tags=["Models"])
async def list_loaded_models_endpoint(
    config: Annotated[ServerConfig, Depends(get_config)]
):
    """
    List currently loaded models (in-memory).
    """
    service = get_transcription_service(config)
    loaded = service.list_loaded_models()
    return JSONResponse(content={"models": loaded})

@router.post("/models/load", tags=["Models"])
async def load_model_endpoint(
    model: Annotated[str, Form()],
    config: Annotated[ServerConfig, Depends(get_config)]
):
    """
    Explicitly load a model into VRAM.
    """
    service = get_transcription_service(config)
    try:
        await service.get_model(model)
        return JSONResponse(content={"status": "success", "model": model})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@router.post("/models/unload", tags=["Models"])
async def unload_model_endpoint(
    model: Annotated[str, Form()],
    config: Annotated[ServerConfig, Depends(get_config)]
):
    """
    Unload a model to free VRAM.
    """
    service = get_transcription_service(config)
    unloaded = await service.unload_model(model)
    if unloaded:
        return JSONResponse(content={"status": "success", "model": model})
    else:
        return JSONResponse(content={"status": "error", "message": f"Model {model} not found or not loaded"}, status_code=404)
