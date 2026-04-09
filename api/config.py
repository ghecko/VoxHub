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
    silero_threshold: float = Field(default=0.35, description="Silero gate sensitivity for hybrid mode")
    override_threshold: float = Field(default=0.8, description="Silero confidence to override Pyannote rejection")

    # Segment post-processing
    min_turn_duration: float = Field(default=1.5, description="Speaker turns shorter than this are absorbed (seconds)")
    refine_boundaries: bool = Field(default=False, description="Use wav2vec2 to snap boundaries to exact speech onset/offset")

    # Transcription settings (from main.py)
    precision: str = "auto"
    flash_attn: bool = False
    compile_model: bool = False

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
