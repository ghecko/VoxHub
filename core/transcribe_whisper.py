import torch
import numpy as np
from typing import Optional, List
from core.base import BaseTranscriber


class WhisperTranscriber(BaseTranscriber):
    """
    Whisper backend using HuggingFace Transformers (openai/whisper-*).
    This avoids the CTranslate2/faster-whisper CUDA compilation issues
    by using the native Transformers pipeline which works out of the box
    on any PyTorch+CUDA environment.
    """

    # Map short names to HuggingFace model IDs
    _HF_MAP = {
        "large-v3": "openai/whisper-large-v3",
        "turbo": "openai/whisper-large-v3-turbo",
        "medium": "openai/whisper-medium",
        "small": "openai/whisper-small",
        "base": "openai/whisper-base",
        "tiny": "openai/whisper-tiny",
    }

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        language: Optional[str] = None,
        **kwargs,
    ):
        hf_id = self._HF_MAP.get(model_size, model_size)
        super().__init__(hf_id, device, language=language)
        self.model = None
        self.processor = None
        self.supports_context_carry = True

    def load(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        device = self.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self._device = device
        self._torch_dtype = torch_dtype

    def transcribe_segment(
        self, 
        audio: np.ndarray, 
        language: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """Transcribe a single audio segment."""
        input_features = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(self.model.device, dtype=self._torch_dtype)

        generate_kwargs = {"max_new_tokens": 440}
        
        # Priority: requested language > model default language
        target_lang = language or self.language
        if target_lang:
            generate_kwargs["language"] = target_lang

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                **generate_kwargs,
            )

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()

    def transcribe_batch(self, audio_segments: List[np.ndarray]) -> List[str]:
        """Sequential fallback — Whisper Transformers doesn't batch easily."""
        return super().transcribe_batch(audio_segments)
