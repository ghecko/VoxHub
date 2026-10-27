from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Dict

class BaseTranscriber(ABC):
    """
    Abstract base class for all ASR backends.
    Unified interface for loading models and transcribing audio segments.
    """
    
    def __init__(self, model_id: str, device: str = "auto", language: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self.language = language
        self.supports_context_carry = False
        # True for backends that can bias decoding towards a list of terms
        # (Mistral's `context_bias`). None of the current ones can: the open
        # Voxtral-Mini-3B-2507 has no such input, and vLLM forwards only the
        # audio and language to Voxtral.
        self.supports_context_bias = False
        # True for HTTP-client backends (vLLM): chunks may be transcribed concurrently.
        self.is_remote = False

    @abstractmethod
    def load(self) -> None:
        """Load the model and processor into memory/device."""
        pass

    @abstractmethod
    def transcribe_segment(
        self, 
        audio: np.ndarray, 
        language: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Transcribe a single audio segment.
        Optionally accepts language and context (previous transcript) for continuity.
        """
        pass

    def transcribe_batch(self, audio_segments: List[np.ndarray]) -> List[str]:
        """
        Transcribe multiple segments in a batch (if supported by backend).
        Default implementation falls back to sequential transcription.
        """
        return [self.transcribe_segment(seg) for seg in audio_segments]

    @staticmethod
    def estimate_max_tokens(audio: np.ndarray) -> int:
        """Estimate max output tokens from audio length (~10 tokens/sec)."""
        duration_s = len(audio) / 16000
        return max(64, int(duration_s * 10))
