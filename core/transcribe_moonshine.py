import torch
import numpy as np
from typing import Optional, List
from transformers import pipeline
from core.base import BaseTranscriber

class MoonshineTranscriber(BaseTranscriber):
    """
    Moonshine backend using Transformers pipeline.
    Useful for quick inference on CPU.
    """
    def __init__(self, model_name: str = "UsefulSensors/moonshine-base", device: str = "auto", **kwargs):
        super().__init__(model_name, device)
        self.pipe = None

    def load(self):
        device = self.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Moonshine is supported via transformers ASR pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=device,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32
        )

    def transcribe_segment(
        self, 
        audio: np.ndarray, 
        language: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Transcribe a segment using Moonshine.
        """
        # Moonshine doesn't natively support text context in the standard ASR pipeline,
        # but we could potentially wrap it differently if needed.
        result = self.pipe({"raw": audio, "sampling_rate": 16000})
        return result["text"].strip()
