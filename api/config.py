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
    NONE = "none"

class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOXBENCH_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore"
    )

    # Core settings
    model: str = Field(default="whisper:turbo")
    device: Device = Field(default=Device.AUTO)
    vad: VadMode = Field(default=VadMode.PYANNOTE)
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
    
    # Transcription settings (from main.py)
    precision: str = "auto"
    flash_attn: bool = False
    compile_model: bool = False

def get_config() -> ServerConfig:
    return ServerConfig()
