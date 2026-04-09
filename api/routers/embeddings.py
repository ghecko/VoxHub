"""
Speaker embedding extraction endpoint.

POST /v1/audio/embeddings — extracts a single speaker embedding vector
from an audio file. Designed for short voice samples (5-30s of a single speaker).
"""

import os
import shutil
import asyncio
import tempfile
import logging
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Form, UploadFile, Request, HTTPException, status
from fastapi.responses import JSONResponse

from api.config import get_config, ServerConfig
from api.middleware import ApiKeyDependency
from core.audio import load_audio
from core.embeddings import extract_embedding_from_audio, validate_single_speaker

logger = logging.getLogger(__name__)
router = APIRouter()


def is_secure(request: Request, config: ServerConfig) -> bool:
    """Check if request came over HTTPS.

    Handles reverse proxy setups where the proxy terminates TLS
    and forwards X-Forwarded-Proto.
    """
    if config.allow_insecure_embeddings:
        return True
    # Direct HTTPS
    if request.url.scheme == "https":
        return True
    # Behind a reverse proxy (Caddy, Nginx, etc.)
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    return forwarded_proto.lower() == "https"


@router.post(
    "/v1/audio/embeddings",
    tags=["Embeddings"],
    description="Extract a speaker embedding vector from an audio file.",
)
async def extract_embedding(
    request: Request,
    file: UploadFile,
    config: Annotated[ServerConfig, Depends(get_config)],
    auth: Annotated[None, ApiKeyDependency] = None,
    model: Annotated[Optional[str], Form()] = None,
):
    """
    Extract a 512-dimensional speaker embedding from a voice sample.

    The audio should contain a single speaker (5-30 seconds recommended).
    Returns an L2-normalized embedding vector suitable for speaker identification.
    """
    # ── HTTPS enforcement ──────────────────────────────────────
    if not is_secure(request, config):
        raise HTTPException(
            status_code=403,
            detail=(
                "Embedding extraction requires HTTPS. "
                "Refusing to transmit biometric data over an unencrypted channel."
            ),
        )

    request_id = getattr(request.state, "request_id", "unknown")

    # ── Save uploaded file ─────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        # ── Load audio ─────────────────────────────────────────
        logger.info(f"[{request_id}] Loading audio for embedding extraction: {file.filename}")
        audio = await asyncio.to_thread(load_audio, temp_path)
        duration = len(audio) / 16000.0

        # ── Validate minimum duration ──────────────────────────
        if duration < 1.0:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short ({duration:.1f}s). Minimum 1 second required for embedding extraction.",
            )

        # ── Optional: validate single speaker ──────────────────
        try:
            is_single, speaker_count = await asyncio.to_thread(
                validate_single_speaker,
                audio,
                16000,
                config.hf_token,
            )
            if not is_single:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Multiple speakers detected ({speaker_count}) in the audio sample. "
                        "Please provide a recording with only one speaker."
                    ),
                )
        except HTTPException:
            raise
        except Exception as e:
            # If validation fails (e.g. model not available), log and continue
            logger.warning(f"[{request_id}] Single-speaker validation skipped: {e}")

        # ── Extract embedding ──────────────────────────────────
        logger.info(f"[{request_id}] Extracting speaker embedding ({duration:.1f}s audio)")
        embedding = await asyncio.to_thread(
            extract_embedding_from_audio,
            audio,
            16000,
            config.hf_token,
        )

        model_name = model or "pyannote/embedding"
        return JSONResponse(content={
            "embedding": embedding,
            "embedding_dim": len(embedding),
            "model": model_name,
            "duration": round(duration, 1),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Embedding extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding extraction failed: {str(e)}",
        )
    finally:
        # ── Clean up temp file immediately ─────────────────────
        if os.path.exists(temp_path):
            os.remove(temp_path)
