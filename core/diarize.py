import os
import logging
from pyannote.audio import Pipeline
import torch
import numpy as np
import warnings
from typing import List, Dict, Optional, Callable

# Suppress pyannote's verbose torchcodec/ffmpeg loading warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

logger = logging.getLogger(__name__)

# ── Pyannote pipeline step → progress mapping ────────────────────────
# The pyannote diarization pipeline runs several internal steps.  We map
# known step names to progress percentages within the diarization phase
# so the caller can display a meaningful progress bar.
_PYANNOTE_STEP_PROGRESS: Dict[str, float] = {
    "segmentation": 0.15,
    "embeddings": 0.50,
    "clustering": 0.85,
}


class DiarizationAnalyzer:
    def __init__(self, model_id: str = "pyannote/speaker-diarization-community-1", auth_token: str = None):
        self.pipeline = Pipeline.from_pretrained(
            model_id,
            token=auth_token or os.getenv("HF_TOKEN")
        )
        if torch.cuda.is_available():
            # For ROCm, torch.cuda.is_available() is True, and 'cuda' refers to the ROCm device
            self.pipeline.to(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            self.pipeline.to(torch.device("mps"))
        # The pipeline clusters with its own embedding model; knowing which one
        # matters when choosing VOXHUB_EMBEDDING_MODEL (same space = free
        # consistency between clustering, merging and voice profiles).
        logger.info("Diarization pipeline %s (embedding model: %s)",
                    model_id, getattr(self.pipeline, "embedding", "unknown"))

    def diarize(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> List[Dict]:
        """
        Diarize preloaded audio and return speaker segments.
        Pyannote's pipeline includes built-in VAD, so no separate VAD step is needed.

        Args:
            on_progress: optional callback(step_name, fraction_0_to_1) called
                         when pyannote enters a new internal processing step.
        """
        # Copy to ensure the array is writable before converting to a torch tensor
        waveform = torch.from_numpy(audio.copy()).unsqueeze(0)

        # Pyannote expects a dictionary for in-memory audio
        input_data = {"waveform": waveform, "sample_rate": sampling_rate}

        # Build kwargs for speaker count hints
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        # If we have a progress callback, create a pyannote hook that translates
        # internal step notifications into progress updates.
        if on_progress is not None:
            def _pyannote_hook(step_name: str, step_artifact, file):
                frac = _PYANNOTE_STEP_PROGRESS.get(step_name, None)
                if frac is not None:
                    logger.debug("[Diarize] pyannote step: %s (%.0f%%)", step_name, frac * 100)
                    on_progress(step_name, frac)

            try:
                output = self.pipeline(input_data, hook=_pyannote_hook, **kwargs)
            except TypeError:
                # Older pyannote versions may not support the `hook` kwarg.
                logger.debug("[Diarize] pyannote hook not supported; falling back")
                output = self.pipeline(input_data, **kwargs)
        else:
            output = self.pipeline(input_data, **kwargs)

        # Handle both old (Annotation) and new (DiarizeOutput) pyannote versions
        if hasattr(output, "speaker_diarization"):
            diarization = output.speaker_diarization
        else:
            diarization = output

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments
