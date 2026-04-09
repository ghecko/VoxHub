import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.config import get_config
from api.middleware import RequestIDMiddleware
from api.transcriber import get_transcription_service
from api.routers import health, models, transcriptions, embeddings

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voxhub-api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    logger.info(f"Starting VoxHub API Server")
    logger.info(f"Default Model: {config.model}")
    logger.info(f"Default VAD  : {config.vad}")
    logger.info(f"Diarization  : {config.diarize}")
    
    # Check for HF Token
    if config.hf_token:
        logger.info("HF_TOKEN detected (for pyannote/gated models)")
    else:
        logger.warning("HF_TOKEN not detected. pyannote VAD/Diarization might fail.")

    # Start the job TTL cleanup loop
    service = get_transcription_service(config)
    service.start_cleanup_loop()
    ttl_display = f"{config.result_ttl}s" if config.result_ttl > 0 else "disabled"
    logger.info(f"Result TTL   : {ttl_display}")

    # Embedding security notice
    if config.allow_insecure_embeddings:
        logger.warning(
            "Embeddings over HTTP enabled (VOXHUB_ALLOW_INSECURE_EMBEDDINGS=true). "
            "This is safe inside a Docker network. If VoxHub is exposed publicly, "
            "set VOXHUB_ALLOW_INSECURE_EMBEDDINGS=false to enforce HTTPS."
        )

    yield
    logger.info("Shutting down VoxHub API Server")

def create_app() -> FastAPI:
    config = get_config()
    app = FastAPI(
        title="VoxHub OpenAI-Compatible API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Middleware
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Routers
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(transcriptions.router)
    app.include_router(embeddings.router)
    
    return app

app = create_app()

if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        reload=False, # Disable reload for production/GPU stability
        workers=1     # Sequential request handling is better for single-GPU setups
    )
