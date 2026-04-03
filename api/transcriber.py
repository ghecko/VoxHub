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
from api.config import ServerConfig

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._models: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._vad_engines: Dict[str, UnifiedVAD] = {}
        self._boundary_refiner: Optional[BoundaryRefiner] = None
        self._jobs: Dict[str, Dict[str, Any]] = {}

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

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def create_job(self, job_id: str):
        self._jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "stage": None,          # Additive field: "loading", "vad", "transcribing", None when done
            "progress": 0,
            "created_at": time.time(),
            "completed_at": None,
            "result": None,
            "error": None
        }

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
        job_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        model_spec = normalize_model_spec(model_spec or self.config.model)
        vad_mode = vad_mode or self.config.vad
        diarize = diarize if diarize is not None else self.config.diarize
        
        async with self._semaphore:
            # 1. Load Audio
            if job_id:
                self._update_job(job_id, stage="loading")
            logger.info(f"[{request_id}] Loading audio: {audio_path}")
            audio = await asyncio.to_thread(load_audio, audio_path)

            # 2. Run VAD/Diarization
            if job_id:
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
                self._update_job(job_id, stage="transcribing")
            transcriber = await self.get_model(model_spec)

            # 4. Transcribe segments
            logger.info(f"[{request_id}] Transcribing {len(segments)} segments with {model_spec}")
            final_data = []
            current_context = prompt # Initial prompt from user
            sampling_rate = 16000
            
            for i, seg in enumerate(segments):
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
            
            if job_id:
                self._update_job(job_id, status="completed", stage=None, progress=100, result=final_data, completed_at=time.time())
                
            return final_data

    async def transcribe_job_runner(
        self,
        job_id: str,
        audio_path: str,
        model_spec: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        vad_mode: Optional[str] = None,
        diarize: Optional[bool] = None,
        request_id: str = ""
    ):
        try:
            self._update_job(job_id, status="processing")
            await self.transcribe(
                audio_path=audio_path,
                model_spec=model_spec,
                language=language,
                prompt=prompt,
                vad_mode=vad_mode,
                diarize=diarize,
                request_id=request_id,
                job_id=job_id
            )
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
