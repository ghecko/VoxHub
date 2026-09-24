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
from core.lang_detect import WhisperLanguageDetector, validate_detected_language
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
        self._aligner = None                 # core.align.ForcedAligner (wordalign)
        self._aligner_failed = False
        self._chunk_vad = None               # SileroVAD used for chunk boundaries
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
            "stage": None,          # "loading", "detecting_language", "vad", "diarizing", "transcribing", "aligning", "embeddings", None when done
            "progress": 0,
            "chunks_done": None,    # wordalign only: transcription sub-progress
            "chunks_total": None,
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

    # ── Pipeline progress ranges ────────────────────────────────────
    # Each pipeline phase occupies a fixed slice of the 0–100 progress
    # bar.  Frequent updates within each phase keep the stale-job timer
    # in polling consumers (OpenHiNotes) happy and give users meaningful
    # feedback about what is happening.
    _PROG_LOADING      = (0,  5)    # audio load
    _PROG_LANG_DETECT  = (5,  8)    # language detection
    # legacy pipeline
    _PROG_VAD          = (8, 45)    # VAD + diarization
    _PROG_SANITIZE     = (45, 48)   # segment sanitisation / boundary refinement
    _PROG_MODEL_LOAD   = (48, 51)   # transcription model load
    _PROG_TRANSCRIBE   = (52, 95)   # segment-by-segment transcription
    # wordalign pipeline (transcription and diarization run concurrently)
    _PROG_WA_CHUNK      = (8, 10)   # Silero chunking
    _PROG_WA_TRANSCRIBE = (10, 60)  # chunk transcription (∥ diarization)
    _PROG_WA_DIARIZE    = (60, 70)  # waiting for diarization to finish
    _PROG_WA_ALIGN      = (70, 92)  # CTC alignment + speaker assignment
    _PROG_EMBEDDINGS   = (95, 100)  # speaker embedding extraction
    # NOTE: the 1-point gap between MODEL_LOAD end (51) and TRANSCRIBE
    # start (52) is intentional — it guarantees a progress change when
    # entering the transcription loop, which resets the stale-job timer
    # in polling consumers.

    _NATIVE_LID_PREFIXES = ("whisper:", "voxtral:")

    def _job_progress(self, job_id: Optional[str], progress: float,
                      stage: Optional[str] = None, **extra):
        """Convenience helper: update job progress (and optionally stage)."""
        if not job_id:
            return
        kwargs: Dict[str, Any] = {"progress": round(progress, 1)}
        if stage is not None:
            kwargs["stage"] = stage
        kwargs.update(extra)
        self._update_job(job_id, **kwargs)

    @staticmethod
    def _lerp(rng: tuple, frac: float) -> float:
        """Linearly interpolate within a (lo, hi) progress range."""
        lo, hi = rng
        return lo + (hi - lo) * max(0.0, min(1.0, frac))

    def _check_cancelled(self, job_id: Optional[str], where: str):
        if job_id and self._is_cancelled(job_id):
            raise CancelledError(f"Job {job_id} cancelled {where}")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_aligner(self):
        """Lazy-load the CTC forced aligner (wordalign pipeline)."""
        if self._aligner is None and not self._aligner_failed:
            try:
                from core.align import ForcedAligner
                device = self.config.device.value if hasattr(self.config.device, "value") else str(self.config.device)
                if device in ("auto", "rocm"):
                    device = "auto"
                self._aligner = ForcedAligner(model=self.config.align_model, device=device)
            except Exception as e:
                self._aligner_failed = True
                logger.error(
                    "Could not load the forced aligner (%s): %s. "
                    "Falling back to the legacy pipeline for this process.",
                    self.config.align_model, e,
                )
        return self._aligner

    def _get_chunk_vad(self):
        """Silero instance used for silence-bounded chunking (wordalign)."""
        if self._chunk_vad is None:
            from core.vad import SileroVAD
            self._chunk_vad = SileroVAD(
                threshold=self.config.chunk_silero_threshold,
                min_speech_duration_ms=200,
                min_silence_duration_ms=300,
            )
        return self._chunk_vad

    async def _detect_language(self, audio, model_spec: str, language: Optional[str],
                               request_id: str, job_id: Optional[str]) -> tuple:
        """Return (language_hint_for_backend, detected_language_for_response).

        Backends with native LID (whisper, voxtral) never receive a hint we
        derived from the probe — they are better at it than whisper-base —
        but we still report the detected code in the response so consumers
        get a real ``language`` field instead of "unknown".
        """
        normalized = (language or "").strip().lower()
        if normalized not in ("", "auto"):
            return normalized, normalized

        detector = self._get_lang_detector()
        if detector is None:
            return None, None
        self._job_progress(job_id, self._PROG_LANG_DETECT[0], stage="detecting_language")
        try:
            detected = await asyncio.to_thread(detector.detect, audio)
        except Exception as e:
            logger.warning(f"[{request_id}] Language detection failed: {e}")
            return None, None
        if not detected:
            logger.info(f"[{request_id}] Language detection inconclusive; proceeding without hint")
            return None, None

        if model_spec.startswith(self._NATIVE_LID_PREFIXES):
            logger.info(f"[{request_id}] Detected language {detected} (native-LID backend, hint not forwarded)")
            return None, detected
        validated = validate_detected_language(detected, model_spec)
        if validated:
            logger.info(f"[{request_id}] Auto-detected language: {detected}")
            return validated, detected
        logger.info(
            f"[{request_id}] Detected '{detected}' not supported by {model_spec}; "
            f"proceeding without language hint"
        )
        return None, detected

    async def _run_diarization(self, audio, vad_mode: str, diarize: bool, request_id: str,
                               job_id: Optional[str], prog_range: tuple,
                               speaker_hints: Dict[str, Any]) -> List[Dict]:
        """Run the VAD/diarization engine in a thread with a keepalive heartbeat.

        Returns the raw segment list (with ``speaker`` when diarize=True).
        """
        vad_engine = self.get_vad(vad_mode)

        def _vad_progress(stage_name: str, frac: float):
            self._job_progress(job_id, self._lerp(prog_range, frac), stage=stage_name)

        vad_future = asyncio.ensure_future(
            asyncio.to_thread(
                vad_engine.detect,
                audio,
                diarize=diarize,
                on_progress=_vad_progress,
                **speaker_hints,
            )
        )

        # The pyannote community pipeline doesn't report internal step
        # progress, so the bar would otherwise freeze for the entire
        # diarization. Bump progress by +0.1 every few seconds — just enough
        # to reset the stale-job timer in OpenHiNotes without lying about
        # real progress.
        _HEARTBEAT_INTERVAL = 5
        _HEARTBEAT_BUMP = 0.1
        ceiling = prog_range[1] - 1
        while not vad_future.done():
            try:
                await asyncio.wait_for(asyncio.shield(vad_future), timeout=_HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                pass
            if not vad_future.done() and job_id:
                cur = self._jobs.get(job_id, {}).get("progress", 0)
                self._job_progress(
                    job_id, min(cur + _HEARTBEAT_BUMP, ceiling),
                    stage="diarizing" if diarize else "vad",
                )
        return vad_future.result()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

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
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        pipeline: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe *audio_path* and return a result dict::

            {
              "segments": [{id, start, end, speaker, text, confidence?, words?}, ...],
              "language": "fr" | None,
              "duration": 1234.5,          # real audio duration in seconds
              "pipeline": "wordalign" | "legacy",
              "warnings": [...],
              "speaker_embeddings": {...}, # only when requested
              "speaker_embedding_model": {"id": ..., "dim": ...},
            }
        """
        model_spec = normalize_model_spec(model_spec or self.config.model)
        vad_mode = str(vad_mode or (self.config.vad.value if hasattr(self.config.vad, "value") else self.config.vad))
        diarize = diarize if diarize is not None else self.config.diarize
        pipeline = str(pipeline or (self.config.pipeline.value if hasattr(self.config.pipeline, "value") else self.config.pipeline))
        speaker_hints = {
            k: v for k, v in (
                ("num_speakers", num_speakers),
                ("min_speakers", min_speakers),
                ("max_speakers", max_speakers),
            ) if v is not None
        }
        warnings: List[str] = []

        async with self._semaphore:
            # ── 1. Load audio ─────────────────────────────────────────
            self._check_cancelled(job_id, "before loading")
            self._job_progress(job_id, self._PROG_LOADING[0], stage="loading")
            logger.info(f"[{request_id}] Loading audio: {audio_path}")
            audio = await asyncio.to_thread(load_audio, audio_path)
            duration = len(audio) / 16000.0
            self._job_progress(job_id, self._PROG_LOADING[1])

            # ── 1a. Fail fast on the ASR backend ──────────────────────
            # For remote backends (vLLM) this is a cheap /models probe; for
            # in-process models it loads the weights, which we need anyway.
            # Doing it here means an unreachable vllm service is reported in
            # ~1 s instead of after language detection, aligner load and
            # chunking (~30-60 s on a cold start).
            self._job_progress(job_id, self._PROG_LOADING[1], stage="loading_model")
            await self.get_model(model_spec)

            # ── 1b. Language detection ────────────────────────────────
            lang_hint, detected_language = await self._detect_language(
                audio, model_spec, language, request_id, job_id
            )
            self._job_progress(job_id, self._PROG_LANG_DETECT[1])

            # ── 2-4. Pipeline ─────────────────────────────────────────
            # Aligner load (~1.2 GB for MMS_FA on first use) happens off the loop.
            if pipeline == "wordalign" and await asyncio.to_thread(self._get_aligner) is None:
                warnings.append(
                    f"forced aligner '{self.config.align_model}' unavailable; "
                    "fell back to the legacy pipeline"
                )
                pipeline = "legacy"

            if pipeline == "wordalign":
                final_data = await self._run_wordalign(
                    audio, model_spec, lang_hint, prompt, diarize,
                    request_id, job_id, speaker_hints, warnings,
                )
            else:
                if diarize and vad_mode == "silero":
                    warnings.append(
                        "diarize=true with vad_mode=silero: Silero has no speaker "
                        "labels, every segment is SPEAKER_00. Use vad_mode=pyannote "
                        "or hybrid (or the wordalign pipeline) for diarization."
                    )
                final_data = await self._run_legacy(
                    audio, model_spec, lang_hint, prompt, vad_mode, diarize,
                    request_id, job_id, speaker_hints,
                )

            result: Dict[str, Any] = {
                "segments": final_data,
                "language": detected_language,
                "duration": round(duration, 3),
                "pipeline": pipeline,
                "warnings": warnings,
            }

            # ── 5. Speaker embeddings ─────────────────────────────────
            if return_speaker_embeddings and diarize and final_data:
                self._job_progress(job_id, self._PROG_EMBEDDINGS[0], stage="embeddings")
                try:
                    from core.embeddings import (
                        extract_per_speaker_embeddings, EMBEDDING_MODEL_ID, EMBEDDING_DIM,
                    )
                    logger.info(f"[{request_id}] Extracting per-speaker embeddings")
                    result["speaker_embeddings"] = await asyncio.to_thread(
                        extract_per_speaker_embeddings,
                        audio,
                        final_data,
                        16000,
                        self.config.hf_token,
                    )
                    result["speaker_embedding_model"] = {"id": EMBEDDING_MODEL_ID, "dim": EMBEDDING_DIM}
                except Exception as e:
                    logger.warning(f"[{request_id}] Speaker embedding extraction failed: {e}")
                    warnings.append(f"speaker embedding extraction failed: {e}")

            if job_id:
                self._update_job(
                    job_id, status="completed", stage=None,
                    progress=100, result=result, completed_at=time.time(),
                )
            return result

    # ------------------------------------------------------------------
    # Pipeline A — legacy (diarize → transcribe each turn)
    # ------------------------------------------------------------------

    async def _run_legacy(
        self, audio, model_spec: str, language: Optional[str], prompt: Optional[str],
        vad_mode: str, diarize: bool, request_id: str, job_id: Optional[str],
        speaker_hints: Dict[str, Any],
    ) -> List[Dict]:
        # ── 2. VAD / Diarization ──────────────────────────────────────
        self._check_cancelled(job_id, "before VAD")
        self._job_progress(job_id, self._PROG_VAD[0], stage="vad")
        logger.info(f"[{request_id}] Running VAD ({vad_mode}, diarize={diarize})")
        segments = await self._run_diarization(
            audio, vad_mode, diarize, request_id, job_id, self._PROG_VAD, speaker_hints
        )

        # 2b. Sanitize segments (overlap resolution, micro-turn absorption)
        self._job_progress(job_id, self._PROG_SANITIZE[0])
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
                segments = await asyncio.to_thread(refiner.refine_boundaries, audio, segments)
                logger.info(f"[{request_id}] Boundaries refined with wav2vec2")
        self._job_progress(job_id, self._PROG_SANITIZE[1])

        # ── 3. Load transcription model ───────────────────────────────
        self._check_cancelled(job_id, "before transcription")
        self._job_progress(job_id, self._PROG_MODEL_LOAD[0], stage="transcribing")
        transcriber = await self.get_model(model_spec)
        self._job_progress(job_id, self._PROG_MODEL_LOAD[1], stage="transcribing")

        # ── 4. Transcribe segments ────────────────────────────────────
        logger.info(f"[{request_id}] Transcribing {len(segments)} segments with {model_spec}")
        final_data: List[Dict] = []
        current_context = prompt  # Initial prompt from user
        sampling_rate = 16000
        n_segs = max(len(segments), 1)

        for i, seg in enumerate(segments):
            self._check_cancelled(job_id, f"during transcription (segment {i}/{len(segments)})")

            # Report the START of this segment's slice so the bar reflects
            # "working on segment i" rather than jumping ahead. Critical for
            # single-segment jobs where the only inference call can take minutes.
            self._job_progress(job_id, self._lerp(self._PROG_TRANSCRIBE, i / n_segs))

            start_samp = int(seg["start"] * sampling_rate)
            end_samp = int(seg["end"] * sampling_rate)
            seg_duration = seg["end"] - seg["start"]

            # Drop segments shorter than the configured floor. Below ~0.5s the
            # ASR backend has too little signal to override its language-prior
            # fallback and tends to emit hallucinated boilerplate.
            if seg_duration < self.config.min_segment_duration:
                logger.debug(
                    f"[{request_id}] Skipping short segment "
                    f"[{seg['start']:.2f}-{seg['end']:.2f}s] "
                    f"({seg_duration*1000:.0f}ms < "
                    f"{self.config.min_segment_duration*1000:.0f}ms)"
                )
                continue

            segment_audio = audio[start_samp:end_samp]

            # Energy gate: VAD answers "is there speech somewhere in this
            # window", not "is the SNR high enough for ASR".
            if self.config.min_segment_rms > 0 and segment_audio.size > 0:
                rms = float(np.sqrt(np.mean(segment_audio.astype(np.float32) ** 2)))
                if rms < self.config.min_segment_rms:
                    logger.debug(
                        f"[{request_id}] Skipping low-energy segment "
                        f"[{seg['start']:.2f}-{seg['end']:.2f}s] "
                        f"rms={rms:.4f} < {self.config.min_segment_rms}"
                    )
                    continue

            speaker = seg.get("speaker", "SPEAKER_00")

            # Handle context carry for models that support it
            context = None
            if transcriber.supports_context_carry:
                if i == 0 and prompt:
                    context = prompt
                elif final_data and final_data[-1]["speaker"] == speaker:
                    context = current_context

            text = await asyncio.to_thread(
                transcriber.transcribe_segment,
                segment_audio,
                language=language,
                context=context,
            )

            if i % 5 == 0 or i == len(segments) - 1:
                logger.info(f"[{request_id}] Progress: {i+1}/{len(segments)} segments processed")
            self._job_progress(job_id, self._lerp(self._PROG_TRANSCRIBE, (i + 1) / n_segs))

            if not text or not text.strip():
                continue

            current_context = text

            # Merge if same speaker and small gap (0.8s)
            should_merge = (
                final_data
                and final_data[-1]["speaker"] == speaker
                and (seg["start"] - final_data[-1]["end"]) < 0.8
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
                    "text": str(text),
                })
        return final_data

    # ------------------------------------------------------------------
    # Pipeline B — wordalign (transcribe chunks ∥ diarize → align → assign)
    # ------------------------------------------------------------------

    async def _run_wordalign(
        self, audio, model_spec: str, language: Optional[str], prompt: Optional[str],
        diarize: bool, request_id: str, job_id: Optional[str],
        speaker_hints: Dict[str, Any], warnings: List[str],
    ) -> List[Dict]:
        from core.chunking import build_chunks, split_chunk
        from core.align import assign_word_speakers, words_to_segments, uniform_word_times

        sampling_rate = 16000
        duration = len(audio) / sampling_rate

        # ── 2. Silence-bounded chunking (Silero, CPU, fast) ───────────
        self._check_cancelled(job_id, "before chunking")
        self._job_progress(job_id, self._PROG_WA_CHUNK[0], stage="vad")
        speech = await asyncio.to_thread(self._get_chunk_vad().detect, audio, sampling_rate)
        chunks = build_chunks(
            speech, duration,
            target_duration=self.config.chunk_target_duration,
            max_duration=self.config.chunk_max_duration,
            min_duration=self.config.chunk_min_duration,
        )
        logger.info(
            f"[{request_id}] {len(speech)} speech regions → {len(chunks)} chunks "
            f"(target {self.config.chunk_target_duration:.0f}s, max {self.config.chunk_max_duration:.0f}s)"
        )
        self._job_progress(job_id, self._PROG_WA_CHUNK[1], chunks_total=len(chunks), chunks_done=0)
        if not chunks:
            logger.warning(f"[{request_id}] No speech detected")
            return []

        # ── 3. Diarization in the background ──────────────────────────
        # pyannote needs the *whole* file for consistent speaker labels, so
        # it runs once on the full audio, concurrently with transcription.
        diar_task = None
        if diarize:
            diar_task = asyncio.ensure_future(
                asyncio.to_thread(
                    self.get_vad("pyannote").detect,
                    audio, diarize=True, **speaker_hints,
                )
            )

        # ── 4. Transcribe chunks (concurrently for remote backends) ───
        self._check_cancelled(job_id, "before transcription")
        self._job_progress(job_id, self._PROG_WA_TRANSCRIBE[0], stage="transcribing")
        transcriber = await self.get_model(model_spec)
        is_remote = getattr(transcriber, "is_remote", "vllm" in model_spec)
        concurrency = self.config.transcribe_concurrency if is_remote else 1
        sem = asyncio.Semaphore(concurrency)

        texts: Dict[int, str] = {}          # chunk index → text
        done_count = [0]
        total = [len(chunks)]

        def _context_for(idx: int) -> Optional[str]:
            if not transcriber.supports_context_carry:
                return None
            if idx == 0:
                return prompt
            # Latest finished chunk before this one (may be idx-1 or earlier
            # when running concurrently); the user prompt is the fallback.
            for k in range(idx - 1, -1, -1):
                if texts.get(k):
                    return texts[k]
            return prompt

        async def _transcribe_chunk(chunk: Dict, depth: int = 0) -> str:
            self._check_cancelled(job_id, "during transcription")
            s = int(chunk["start"] * sampling_rate)
            e = int(chunk["end"] * sampling_rate)
            async with sem:
                text = await asyncio.to_thread(
                    transcriber.transcribe_segment,
                    audio[s:e],
                    language=language,
                    context=_context_for(chunk["index"]),
                )
            text = (text or "").strip()
            chunk_len = chunk["end"] - chunk["start"]
            # An empty result on a long chunk is not silence (Silero put speech
            # there): it's a dropped hallucination loop or a timeout. Retry on
            # two halves instead of losing minutes of speech.
            if not text and chunk_len >= 2 * self.config.chunk_min_duration and depth < 2:
                halves = split_chunk(chunk, speech, min_part=self.config.chunk_min_duration / 2)
                if len(halves) == 2:
                    logger.warning(
                        f"[{request_id}] Empty/dropped output for chunk "
                        f"[{chunk['start']:.1f}-{chunk['end']:.1f}s]; retrying on two halves"
                    )
                    total[0] += 1
                    parts = [await _transcribe_chunk(h, depth + 1) for h in halves]
                    text = " ".join(p for p in parts if p).strip()
                    return text
            done_count[0] += 1
            self._job_progress(
                job_id,
                self._lerp(self._PROG_WA_TRANSCRIBE, done_count[0] / max(total[0], 1)),
                chunks_done=done_count[0], chunks_total=total[0],
            )
            if done_count[0] % 5 == 0 or done_count[0] == total[0]:
                logger.info(f"[{request_id}] Progress: {done_count[0]}/{total[0]} chunks transcribed")
            return text

        async def _run(chunk: Dict):
            texts[chunk["index"]] = await _transcribe_chunk(chunk)

        await asyncio.gather(*(_run(c) for c in chunks))

        # ── 5. Wait for diarization ───────────────────────────────────
        turns: List[Dict] = []
        if diar_task is not None:
            self._job_progress(job_id, self._PROG_WA_DIARIZE[0], stage="diarizing")
            while not diar_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(diar_task), timeout=5)
                except asyncio.TimeoutError:
                    pass
                if not diar_task.done() and job_id:
                    cur = self._jobs.get(job_id, {}).get("progress", 0)
                    self._job_progress(job_id, min(cur + 0.1, self._PROG_WA_DIARIZE[1] - 1), stage="diarizing")
            try:
                turns = diar_task.result()
            except Exception as e:
                logger.exception(f"[{request_id}] Diarization failed: {e}")
                warnings.append(f"diarization failed, single speaker assumed: {e}")
                turns = []
            logger.info(f"[{request_id}] Diarization: {len(turns)} turns, "
                        f"{len({t.get('speaker') for t in turns})} speakers")
        self._job_progress(job_id, self._PROG_WA_DIARIZE[1])

        # ── 6. Forced alignment → words ───────────────────────────────
        self._check_cancelled(job_id, "before alignment")
        self._job_progress(job_id, self._PROG_WA_ALIGN[0], stage="aligning")
        aligner = self._get_aligner()
        words: List[Dict] = []
        n_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            text = texts.get(chunk["index"], "")
            if not text:
                continue
            s = int(chunk["start"] * sampling_rate)
            e = int(chunk["end"] * sampling_rate)
            if aligner is not None:
                chunk_words = await asyncio.to_thread(aligner.align, audio[s:e], text, chunk["start"])
            else:
                chunk_words = uniform_word_times(text.split(), chunk["start"], chunk["end"])
            words.extend(chunk_words)
            self._job_progress(job_id, self._lerp(self._PROG_WA_ALIGN, 0.9 * (i + 1) / n_chunks))
        words.sort(key=lambda w: w["start"])

        # ── 7. Word → speaker, words → segments ───────────────────────
        words = assign_word_speakers(words, turns, max_gap=self.config.word_speaker_max_gap)
        segments = words_to_segments(
            words,
            max_pause=self.config.segment_max_pause,
            max_duration=self.config.segment_max_duration,
        )
        self._job_progress(job_id, self._PROG_WA_ALIGN[1])
        logger.info(f"[{request_id}] wordalign: {len(words)} words → {len(segments)} segments")
        return segments

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
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        pipeline: Optional[str] = None,
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
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                pipeline=pipeline,
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
