import os
import time
import torch
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from core.audio import load_audio
from core.registry import create_transcriber, list_supported_models, normalize_model_spec
from core.vad import UnifiedVAD
from core.segments import sanitize_segments, BoundaryRefiner
from core.lang_detect import WhisperLanguageDetector
from api.config import ServerConfig

logger = logging.getLogger(__name__)


class CancelledError(Exception):
    """Raised when a running job is cancelled."""


class TranscriptionService:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._models: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._vad_engines: Dict[str, UnifiedVAD] = {}
        self._boundary_refiner: Optional[BoundaryRefiner] = None
        self._lang_detector: Optional[WhisperLanguageDetector] = None
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._cancel_flags: Dict[str, asyncio.Event] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    def _get_lang_detector(self) -> Optional[WhisperLanguageDetector]:
        """Lazy-construct the Whisper-based language detector."""
        if not getattr(self.config, "auto_detect_language", True):
            return None
        if self._lang_detector is None:
            self._lang_detector = WhisperLanguageDetector(
                model_id=getattr(self.config, "lang_detect_model", "openai/whisper-tiny"),
                device=str(self.config.device.value if hasattr(self.config.device, "value") else self.config.device),
            )
        return self._lang_detector

    def _get_boundary_refiner(self) -> Optional[BoundaryRefiner]:
        """Lazy-load the wav2vec2 boundary refiner."""
        if self._boundary_refiner is None:
            try:
                self._boundary_refiner = BoundaryRefiner(device=self.config.device)
            except Exception as e:
                logger.warning(f"Could not load BoundaryRefiner: {e}")
                return None
        return self._boundary_refiner

    def get_vad(self, mode: str) -> UnifiedVAD:
        if mode not in self._vad_engines:
            self._vad_engines[mode] = UnifiedVAD(
                mode=mode,
                hf_token=self.config.hf_token,
                silero_threshold=self.config.silero_threshold,
                override_threshold=self.config.override_threshold,
            )
        return self._vad_engines[mode]
        
    async def get_model(self, model_spec: str):
        async with self._lock:
            if model_spec not in self._models:
                logger.info(f"Loading model: {model_spec}")
                # Transcriber.load() is blocking, run in thread
                model_kwargs = {}
                if "voxtral" in model_spec:
                    model_kwargs = {
                        "precision": self.config.precision,
                        "flash_attn": self.config.flash_attn,
                        "compile_model": self.config.compile_model
                    }
                
                transcriber = await asyncio.to_thread(
                    create_transcriber,
                    model_spec,
                    device=self.config.device,
                    **model_kwargs
                )
                await asyncio.to_thread(transcriber.load)
                self._models[model_spec] = transcriber
            return self._models[model_spec]

    async def unload_model(self, model_spec: str) -> bool:
        async with self._lock:
            if model_spec in self._models:
                logger.info(f"Unloading model: {model_spec}")
                del self._models[model_spec]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
            return False

    def list_loaded_models(self) -> List[str]:
        return list(self._models.keys())

    # ------------------------------------------------------------------
    # TTL cleanup
    # ------------------------------------------------------------------

    def start_cleanup_loop(self):
        """Start the background task that purges expired jobs. Call once at startup."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.ensure_future(self._ttl_cleanup_loop())

    async def _ttl_cleanup_loop(self):
        """Periodically remove finished jobs older than result_ttl."""
        while True:
            await asyncio.sleep(60)  # check every minute
            self._purge_expired_jobs()

    def _purge_expired_jobs(self):
        ttl = self.config.result_ttl
        if ttl <= 0:
            return  # 0 means keep forever
        now = time.time()
        expired = [
            jid for jid, job in self._jobs.items()
            if job["status"] in ("completed", "failed", "cancelled")
            and job.get("completed_at") is not None
            and (now - job["completed_at"]) > ttl
        ]
        for jid in expired:
            logger.info(f"Purging expired job {jid} (TTL {ttl}s exceeded)")
            self._jobs.pop(jid, None)
            self._cancel_flags.pop(jid, None)

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def list_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all jobs (without result payload), optionally filtered by status.
        Jobs are sorted by created_at descending (newest first)."""
        jobs = []
        for job in self._jobs.values():
            if status_filter and job["status"] != status_filter:
                continue
            jobs.append({k: v for k, v in job.items() if k != "result"})
        jobs.sort(key=lambda j: j["created_at"], reverse=True)
        return jobs

    def create_job(self, job_id: str, return_speaker_embeddings: bool = False):
        self._cancel_flags[job_id] = asyncio.Event()
        self._jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "stage": None,          # Additive field: "loading", "vad", "transcribing", None when done
            "progress": 0,
            "created_at": time.time(),
            "completed_at": None,
            "result": None,
            "error": None,
            "return_speaker_embeddings": return_speaker_embeddings,
        }

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation for a running or pending job.
        Returns True if the cancellation was accepted."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job["status"] in ("completed", "failed", "cancelled"):
            return False  # already terminal
        # Signal the flag so the transcription loop can check it
        flag = self._cancel_flags.get(job_id)
        if flag:
            flag.set()
        # If still pending (not yet picked up), mark immediately
        if job["status"] == "pending":
            self._update_job(job_id, status="cancelled", completed_at=time.time(), result=None)
        return True

    def delete_job(self, job_id: str) -> bool:
        """Remove a finished job from memory. Running jobs must be cancelled first."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job["status"] in ("pending", "processing"):
            return False  # must cancel first
        self._jobs.pop(job_id, None)
        self._cancel_flags.pop(job_id, None)
        return True

    def _is_cancelled(self, job_id: str) -> bool:
        flag = self._cancel_flags.get(job_id)
        return flag is not None and flag.is_set()

    def _update_job(self, job_id: str, **kwargs):
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)

    async def transcribe(
        self,
        audio_path: str,
        model_spec: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        vad_mode: Optional[str] = None,
        diarize: Optional[bool] = None,
        request_id: str = "",
        job_id: Optional[str] = None,
        return_speaker_embeddings: bool = False,
    ):
        model_spec = normalize_model_spec(model_spec or self.config.model)
        vad_mode = vad_mode or self.config.vad
        diarize = diarize if diarize is not None else self.config.diarize
        
        async with self._semaphore:
            # 1. Load Audio
            if job_id:
                if self._is_cancelled(job_id):
                    raise CancelledError(f"Job {job_id} cancelled before loading")
                self._update_job(job_id, stage="loading")
            logger.info(f"[{request_id}] Loading audio: {audio_path}")
            audio = await asyncio.to_thread(load_audio, audio_path)

            # 1b. Optional language auto-detect.
            #   - Skipped if the request supplied an explicit language (other than "auto").
            #   - Skipped for backends that detect natively:
            #       * whisper:*  — Whisper's own decoder picks the language.
            #       * voxtral:*  — Mistral confirms native auto-detection
            #                      across all 13 supported languages.
            #   - On failure we silently proceed with language=None.
            _NATIVE_LID_PREFIXES = ("whisper:", "voxtral:")
            normalized_lang = (language or "").strip().lower()
            needs_detection = (
                normalized_lang in ("", "auto")
                and not model_spec.startswith(_NATIVE_LID_PREFIXES)
            )
            if needs_detection:
                detector = self._get_lang_detector()
                if detector is not None:
                    if job_id:
                        self._update_job(job_id, stage="detecting_language")
                    detected = await asyncio.to_thread(detector.detect, audio)
                    if detected:
                        logger.info(
                            f"[{request_id}] Auto-detected language: {detected}"
                        )
                        language = detected
                    else:
                        logger.info(
                            f"[{request_id}] Language detection inconclusive; "
                            f"proceeding without hint"
                        )

            # 2. Run VAD/Diarization
            if job_id:
                if self._is_cancelled(job_id):
                    raise CancelledError(f"Job {job_id} cancelled before VAD")
                self._update_job(job_id, stage="vad")
            logger.info(f"[{request_id}] Running VAD ({vad_mode}, diarize={diarize})")
            vad_engine = self.get_vad(vad_mode)
            segments = await asyncio.to_thread(
                vad_engine.detect,
                audio,
                diarize=diarize
            )

            # 2b. Sanitize segments (overlap resolution, micro-turn absorption)
            if len(segments) > 1:
                segments = await asyncio.to_thread(
                    sanitize_segments, segments,
                    min_turn_duration=self.config.min_turn_duration,
                )
                logger.info(f"[{request_id}] {len(segments)} segments after sanitization")

            # 2c. Optional wav2vec2 boundary refinement
            if self.config.refine_boundaries:
                refiner = self._get_boundary_refiner()
                if refiner:
                    segments = await asyncio.to_thread(
                        refiner.refine_boundaries, audio, segments
                    )
                    logger.info(f"[{request_id}] Boundaries refined with wav2vec2")

            # 3. Get Model
            if job_id:
                if self._is_cancelled(job_id):
                    raise CancelledError(f"Job {job_id} cancelled before transcription")
                self._update_job(job_id, stage="transcribing")
            transcriber = await self.get_model(model_spec)

            # 4. Transcribe segments
            logger.info(f"[{request_id}] Transcribing {len(segments)} segments with {model_spec}")
            final_data = []
            current_context = prompt # Initial prompt from user
            sampling_rate = 16000
            
            for i, seg in enumerate(segments):
                # Check cancellation between segments
                if job_id and self._is_cancelled(job_id):
                    raise CancelledError(f"Job {job_id} cancelled during transcription (segment {i}/{len(segments)})")

                if i % 5 == 0 or i == len(segments) - 1:
                    logger.info(f"[{request_id}] Progress: {i+1}/{len(segments)} segments processed")
                    if job_id:
                        self._update_job(job_id, progress=round((i+1)/len(segments) * 100, 1))
                
                start_samp = int(seg["start"] * sampling_rate)
                end_samp = int(seg["end"] * sampling_rate)
                duration = seg["end"] - seg["start"]

                if duration < 0.2: # Skip too short segments
                    continue

                segment_audio = audio[start_samp:end_samp]
                speaker = seg.get("speaker", "SPEAKER_00")

                # Handle context carry for models that support it
                context = None
                if transcriber.supports_context_carry:
                    if i == 0 and prompt:
                        context = prompt
                    elif final_data and final_data[-1]["speaker"] == speaker:
                        # Continue from previous segment by this speaker
                        context = current_context

                # Run inference in thread
                text = await asyncio.to_thread(
                    transcriber.transcribe_segment,
                    segment_audio,
                    language=language,
                    context=context
                )

                if not text or not text.strip():
                    continue

                current_context = text

                # Robust Merging Logic: 
                # Merge if same speaker and small gap (0.8s)
                should_merge = (
                    final_data and 
                    final_data[-1]["speaker"] == speaker and 
                    (seg["start"] - final_data[-1]["end"]) < 0.8
                )

                if should_merge:
                    final_data[-1]["end"] = round(seg["end"], 3)
                    final_data[-1]["text"] += " " + str(text)
                else:
                    final_data.append({
                        "id": len(final_data),
                        "start": round(seg["start"], 3),
                        "end": round(seg["end"], 3),
                        "speaker": speaker,
                        "text": str(text)
                    })
            
            # 5. Extract per-speaker embeddings if requested
            result = final_data
            if return_speaker_embeddings and diarize and final_data:
                try:
                    from core.embeddings import extract_per_speaker_embeddings
                    logger.info(f"[{request_id}] Extracting per-speaker embeddings")
                    speaker_embeddings = await asyncio.to_thread(
                        extract_per_speaker_embeddings,
                        audio,
                        final_data,
                        16000,
                        self.config.hf_token,
                    )
                    result = {
                        "segments": final_data,
                        "speaker_embeddings": speaker_embeddings,
                    }
                except Exception as e:
                    logger.warning(f"[{request_id}] Speaker embedding extraction failed: {e}")
                    # Don't fail the transcription — just omit embeddings
                    result = final_data

            if job_id:
                self._update_job(job_id, status="completed", stage=None, progress=100, result=result, completed_at=time.time())

            return result

    async def transcribe_job_runner(
        self,
        job_id: str,
        audio_path: str,
        model_spec: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        vad_mode: Optional[str] = None,
        diarize: Optional[bool] = None,
        request_id: str = "",
        return_speaker_embeddings: bool = False,
    ):
        try:
            # If cancelled while still pending, skip entirely
            if self._is_cancelled(job_id):
                self._update_job(job_id, status="cancelled", completed_at=time.time(), result=None)
                return
            self._update_job(job_id, status="processing")
            await self.transcribe(
                audio_path=audio_path,
                model_spec=model_spec,
                language=language,
                prompt=prompt,
                vad_mode=vad_mode,
                diarize=diarize,
                request_id=request_id,
                job_id=job_id,
                return_speaker_embeddings=return_speaker_embeddings,
            )
        except CancelledError:
            logger.info(f"[{request_id}] Job {job_id} cancelled")
            self._update_job(job_id, status="cancelled", completed_at=time.time(), result=None)
        except Exception as e:
            logger.exception(f"[{request_id}] Job {job_id} failed: {e}")
            self._update_job(job_id, status="failed", error=str(e), completed_at=time.time())
        finally:
            # Cleanup temp file
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass

_service: Optional[TranscriptionService] = None

def get_transcription_service(config: ServerConfig) -> TranscriptionService:
    global _service
    if _service is None:
        _service = TranscriptionService(config)
    return _service
