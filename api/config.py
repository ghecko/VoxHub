from enum import Enum
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ResponseFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    VTT_JSON = "vtt_json"
    SRT = "srt"
    VTT = "vtt"

class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    AUTO = "auto"

class VadMode(str, Enum):
    SILERO = "silero"
    PYANNOTE = "pyannote"
    HYBRID = "hybrid"
    NONE = "none"

class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOXHUB_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore"
    )

    # Core settings
    model: str = Field(default="whisper:turbo")
    device: Device = Field(default=Device.AUTO)
    vad: VadMode = Field(default=VadMode.HYBRID)
    diarize: bool = Field(default=True)
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    allow_origins: List[str] = ["*"]
    
    # Authentication
    api_key: Optional[str] = None
    
    # Hugging Face token (for pyannote/gated models)
    hf_token: Optional[str] = Field(None, alias="HF_TOKEN")
    
    # Performance
    max_concurrent: int = 1

    # Job result retention (seconds). Completed/failed/cancelled jobs are purged after this delay.
    result_ttl: int = Field(default=3600, description="Seconds to keep finished job results (0 = keep forever)")

    # Hybrid VAD tuning (only used when vad=hybrid)
    # Bumped 0.35 -> 0.5 as part of the anti-hallucination work: 0.35 was
    # tuned for maximum recall (catch every utterance) but let through enough
    # noise-classified-as-speech to consistently trigger LM-prior hallucinations
    # in Voxtral on low-SNR chunks. 0.5 is Silero's documented "balanced"
    # default. The override_threshold safety net (0.8) still rescues
    # high-confidence Silero segments that Pyannote rejects.
    silero_threshold: float = Field(default=0.5, description="Silero gate sensitivity for hybrid mode")
    override_threshold: float = Field(default=0.8, description="Silero confidence to override Pyannote rejection")

    # Segment post-processing
    min_turn_duration: float = Field(default=1.5, description="Speaker turns shorter than this are absorbed (seconds)")
    refine_boundaries: bool = Field(default=False, description="Use wav2vec2 to snap boundaries to exact speech onset/offset")

    # Transcription-time segment filters. These run between VAD and the ASR
    # backend and exist to keep low-information audio chunks from reaching
    # the model — every backend we support (Whisper, Voxtral, …) hallucinates
    # boilerplate ("I'm sorry, I can't help…", "Thanks for watching", …) when
    # fed silence or near-silence, so we drop those segments up-front.
    min_segment_duration: float = Field(
        default=0.5,
        description="Skip VAD segments shorter than this (seconds). "
                    "Below ~0.5s, ASR models tend to fall back to LM priors "
                    "and emit hallucinated boilerplate.",
    )
    min_segment_rms: float = Field(
        default=0.005,
        description="Skip segments whose audio RMS is below this threshold "
                    "(float32 audio in [-1, 1]; 0.005 ≈ -46 dBFS). VAD says "
                    "'speech yes/no'; this catches the residual silence/noise "
                    "that VAD lets through. Set to 0 to disable.",
    )

    # Transcription settings (from main.py)
    precision: str = "auto"
    flash_attn: bool = False
    compile_model: bool = False

    # Automatic language detection. When the request has no `language` (or
    # passes "auto"), VoxHub probes the first 30s with a small Whisper model
    # and forwards the detected ISO code to the active backend. Whisper
    # backends are skipped (they detect internally).
    auto_detect_language: bool = Field(
        default=True,
        description="Auto-detect spoken language for backends that need a hint",
    )
    lang_detect_model: str = Field(
        default="openai/whisper-base",
        description="HF model id used for language probing (base is more accurate than tiny with minimal overhead)",
    )

    # Embedding security
    # Voice embeddings are biometric data. By default, cleartext HTTP is allowed
    # because VoxHub typically runs inside a Docker network (traffic never leaves
    # the host). Set to false if VoxHub is exposed on a public or untrusted network.
    allow_insecure_embeddings: bool = Field(
        default=True,
        description="Allow embedding extraction/return over plain HTTP. "
                    "Safe when VoxHub is only reachable within a Docker network. "
                    "Set to false if exposed publicly."
    )

def get_config() -> ServerConfig:
    return ServerConfig()
