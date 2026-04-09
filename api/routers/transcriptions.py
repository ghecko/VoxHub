import os
import uuid
import shutil
import tempfile
import logging
from typing import Annotated, Literal, List, Optional
from fastapi import APIRouter, Depends, Form, UploadFile, Request, HTTPException, status, BackgroundTasks
from fastapi.responses import Response
from api.config import get_config, ServerConfig, ResponseFormat, VadMode
from api.middleware import ApiKeyDependency
from api.transcriber import get_transcription_service, TranscriptionService
from api.formatters import format_transcription
from api.routers.embeddings import is_secure

logger = logging.getLogger(__name__)
router = APIRouter()


def _attach_speaker_embeddings(
    response: Response,
    result_data: dict,
    response_format: ResponseFormat,
    request: Request,
    config: ServerConfig,
) -> Response:
    """Inject speaker_embeddings into a JSON response, respecting HTTPS rules.

    If the request was made over plain HTTP, the embeddings are stripped and a
    warning is added instead.
    """
    import json
    from fastapi.responses import JSONResponse

    # Only inject into JSON-based formats
    if response_format not in (ResponseFormat.JSON, ResponseFormat.VERBOSE_JSON, ResponseFormat.VTT_JSON):
        return response

    # Parse existing response body
    try:
        body = json.loads(response.body)
    except Exception:
        return response

    # HTTPS enforcement: strip embeddings over plain HTTP
    if not is_secure(request, config):
        body["warning"] = "speaker_embeddings omitted: HTTPS required"
        return JSONResponse(content=body)

    body["speaker_embeddings"] = result_data["speaker_embeddings"]
    return JSONResponse(content=body)

@router.post(
    "/v1/audio/transcriptions",
    tags=["Transcription"],
    description="Transcribe audio files using VoxHub backends in an OpenAI-compatible format."
)
async def transcribe_audio(
    request: Request,
    file: UploadFile,
    config: Annotated[ServerConfig, Depends(get_config)],
    auth: Annotated[None, ApiKeyDependency] = None,
    model: Annotated[str, Form()] = None,
    language: Annotated[Optional[str], Form()] = None,
    prompt: Annotated[Optional[str], Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = ResponseFormat.JSON,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        List[Literal["segment", "word"]],
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    diarize: Annotated[Optional[bool], Form()] = None,
    vad_mode: Annotated[Optional[str], Form()] = None,
    return_speaker_embeddings: Annotated[Optional[str], Form()] = "false",
):
    """
    OpenAI-compatible transcription endpoint.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    service = get_transcription_service(config)
    want_embeddings = (return_speaker_embeddings or "").lower() == "true"

    # 1. Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        # 2. Run transcription process
        final_data = await service.transcribe(
            audio_path=temp_path,
            model_spec=model,
            language=language,
            prompt=prompt,
            diarize=diarize,
            vad_mode=vad_mode,
            request_id=request_id,
            return_speaker_embeddings=want_embeddings,
        )

        # 3. Format result — extract segments list for the formatter
        segments = final_data["segments"] if isinstance(final_data, dict) else final_data
        response = format_transcription(segments, response_format)

        # 4. Attach speaker embeddings if requested (JSON formats only)
        if want_embeddings and isinstance(final_data, dict) and "speaker_embeddings" in final_data:
            response = _attach_speaker_embeddings(
                response, final_data, response_format, request, config
            )

        return response

    except Exception as e:
        logger.exception(f"[{request_id}] Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # 5. Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post(
    "/v1/audio/translations",
    tags=["Transcription"],
    description="Translate audio files. (Currently falls back to transcription)."
)
async def translate_audio(
    request: Request,
    file: UploadFile,
    config: Annotated[ServerConfig, Depends(get_config)],
    model: Annotated[str, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = ResponseFormat.JSON,
):
    """
    OpenAI-compatible translation endpoint. 
    VoxHub backends mostly focus on transcription, so we route to transcribe.
    """
    return await transcribe_audio(
        request=request,
        file=file,
        config=config,
        model=model,
        response_format=response_format
    )

@router.post(
    "/v1/audio/transcriptions/jobs",
    tags=["Jobs"],
    description="Start an asynchronous transcription job."
)
async def create_transcription_job(
    request: Request,
    file: UploadFile,
    background_tasks: BackgroundTasks,
    config: Annotated[ServerConfig, Depends(get_config)],
    auth: Annotated[None, ApiKeyDependency] = None,
    model: Annotated[str, Form()] = None,
    language: Annotated[Optional[str], Form()] = None,
    prompt: Annotated[Optional[str], Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = ResponseFormat.JSON,
    diarize: Annotated[Optional[bool], Form()] = None,
    vad_mode: Annotated[Optional[str], Form()] = None,
    return_speaker_embeddings: Annotated[Optional[str], Form()] = "false",
):
    request_id = getattr(request.state, "request_id", "unknown")
    job_id = str(uuid.uuid4())
    service = get_transcription_service(config)
    want_embeddings = (return_speaker_embeddings or "").lower() == "true"

    # 1. Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    # 2. Create job state (store embedding preference in metadata)
    service.create_job(job_id, return_speaker_embeddings=want_embeddings)

    # 3. Queue background task
    background_tasks.add_task(
        service.transcribe_job_runner,
        job_id=job_id,
        audio_path=temp_path,
        model_spec=model,
        language=language,
        prompt=prompt,
        diarize=diarize,
        vad_mode=vad_mode,
        request_id=request_id,
        return_speaker_embeddings=want_embeddings,
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "links": {
            "status": f"/v1/audio/transcriptions/jobs/{job_id}",
            "result": f"/v1/audio/transcriptions/jobs/{job_id}/result"
        }
    }

@router.get(
    "/v1/audio/transcriptions/jobs",
    tags=["Jobs"],
    description="List all transcription jobs with their current state. Optionally filter by status."
)
async def list_jobs(
    config: Annotated[ServerConfig, Depends(get_config)],
    auth: Annotated[None, ApiKeyDependency] = None,
    status_filter: Optional[str] = None,
):
    """
    Returns an overview of every job in the queue, sorted newest-first.
    Pass ?status=pending|processing|completed|failed|cancelled to filter.
    """
    service = get_transcription_service(config)
    jobs = service.list_jobs(status_filter=status_filter)
    counts = {}
    for j in service.list_jobs():
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    return {
        "jobs": jobs,
        "total": sum(counts.values()),
        "counts": counts,
    }

@router.get(
    "/v1/audio/transcriptions/jobs/{job_id}",
    tags=["Jobs"],
    description="Get the status and progress of a transcription job."
)
async def get_job_status(
    job_id: str,
    config: Annotated[ServerConfig, Depends(get_config)],
    auth: Annotated[None, ApiKeyDependency] = None,
):
    service = get_transcription_service(config)
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Return status without the bulky result
    status_data = {k: v for k, v in job.items() if k != "result"}
    return status_data

@router.post(
    "/v1/audio/transcriptions/jobs/{job_id}/cancel",
    tags=["Jobs"],
    description="Cancel a pending or running transcription job. Results are discarded."
)
async def cancel_job(
    job_id: str,
    config: Annotated[ServerConfig, Depends(get_config)],
    auth: Annotated[None, ApiKeyDependency] = None,
):
    service = get_transcription_service(config)
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    accepted = service.cancel_job(job_id)
    if not accepted:
        raise HTTPException(
            status_code=409,
            detail=f"Job cannot be cancelled. Current status: {job['status']}"
        )

    return {"job_id": job_id, "status": "cancelled"}

@router.delete(
    "/v1/audio/transcriptions/jobs/{job_id}",
    tags=["Jobs"],
    description="Delete a finished job (completed, failed, or cancelled). Running jobs must be cancelled first."
)
async def delete_job(
    job_id: str,
    config: Annotated[ServerConfig, Depends(get_config)],
    auth: Annotated[None, ApiKeyDependency] = None,
):
    service = get_transcription_service(config)
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    deleted = service.delete_job(job_id)
    if not deleted:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete a job that is still running (status: {job['status']}). Cancel it first."
        )

    return {"job_id": job_id, "deleted": True}

@router.get(
    "/v1/audio/transcriptions/jobs/{job_id}/result",
    tags=["Jobs"],
    description="Get the final result of a completed transcription job."
)
async def get_job_result(
    job_id: str,
    request: Request,
    config: Annotated[ServerConfig, Depends(get_config)],
    auth: Annotated[None, ApiKeyDependency] = None,
    response_format: ResponseFormat = ResponseFormat.JSON,
):
    service = get_transcription_service(config)
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job['status']}"
        )

    result = job["result"]
    response = format_transcription(
        result["segments"] if isinstance(result, dict) else result,
        response_format,
    )

    # Attach speaker embeddings if they were computed for this job
    if isinstance(result, dict) and "speaker_embeddings" in result:
        response = _attach_speaker_embeddings(
            response, result, response_format, request, config
        )

    return response
