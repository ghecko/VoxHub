import torch
import numpy as np
import os
import tempfile
import soundfile as sf
from typing import Optional, List
from core.base import BaseTranscriber


class CanaryTranscriber(BaseTranscriber):
    """
    NVIDIA Canary backend using NeMo.
    Requires nemo_toolkit[asr].
    
    Canary is a multi-task model supporting ASR and AST (translation).
    Languages supported: en, fr, de, es (Canary-1B).
    """

    def __init__(self, model_name: str = "nvidia/canary-1b", device: str = "auto", language: Optional[str] = None, **kwargs):
        super().__init__(model_name, device, language=language)
        self.model = None

    def load(self):
        try:
            from nemo.collections.asr.models import EncDecMultiTaskModel
        except ImportError:
            raise ImportError(
                "nemo_toolkit[asr] not found. "
                "Canary models require NeMo. Use the specialized Dockerfile.spark image."
            )

        # Load the Canary model
        self.model = EncDecMultiTaskModel.from_pretrained(self.model_id)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()

        # Configure decoding for faster inference
        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        self.model.change_decoding_strategy(decode_cfg)

    def transcribe_segment(
        self, 
        audio: np.ndarray, 
        language: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Transcribe a segment using Canary.
        Uses the NeMo API with explicit language control.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, 16000)

        try:
            # Priority: requested language > model default language
            lang = language or self.language or "en"

            # Canary transcribe() accepts these keyword arguments
            # for the multi-task prompt tokens:
            #   source_lang, target_lang, pnc (punctuation & capitalization)
            results = self.model.transcribe(
                [tmp_path],
                batch_size=1,
                source_lang=lang,
                target_lang=lang,
                pnc="yes",
            )

            if not results:
                return ""

            res = results[0]

            # Handle nested lists (some NeMo versions)
            if isinstance(res, list) and len(res) > 0:
                res = res[0]

            # Handle Hypothesis objects
            if hasattr(res, "text"):
                return str(res.text).strip()

            return str(res).strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
