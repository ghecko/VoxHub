"""
Speaker embedding extraction using pyannote/embedding.

This module provides utilities for extracting speaker voice fingerprints
from audio. Embeddings are 512-dimensional L2-normalized vectors that can
be used to identify speakers across recordings.

IMPORTANT: VoxHub never persists embeddings. They are computed on-the-fly
and returned in API responses. The consuming application is responsible
for storage and matching.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model holder — loaded once at first use
# ---------------------------------------------------------------------------

_embedding_model = None
_embedding_model_lock = None  # Set at first call (needs event loop context)


def _get_embedding_model(hf_token: Optional[str] = None):
    """Load the pyannote embedding model (lazy singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from pyannote.audio import Model

        token = hf_token or os.getenv("HF_TOKEN")
        logger.info("Loading speaker embedding model: pyannote/embedding")
        _embedding_model = Model.from_pretrained("pyannote/embedding", token=token)

        # Move to best available device
        if torch.cuda.is_available():
            _embedding_model = _embedding_model.to(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            _embedding_model = _embedding_model.to(torch.device("mps"))

        logger.info("Speaker embedding model loaded successfully")
    return _embedding_model


def extract_embedding_from_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hf_token: Optional[str] = None,
) -> List[float]:
    """Extract a single speaker embedding from an audio waveform.

    Args:
        audio: 1-D numpy array of audio samples (mono, float32).
        sample_rate: Sample rate of the audio (default 16000).
        hf_token: HuggingFace token for gated model access.

    Returns:
        List of 512 floats — the L2-normalized speaker embedding.
    """
    from pyannote.audio import Inference

    model = _get_embedding_model(hf_token)
    inference = Inference(model, window="whole")

    waveform = torch.from_numpy(audio.copy()).unsqueeze(0).float()
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    embedding = inference(audio_input)
    return embedding.tolist()


def extract_per_speaker_embeddings(
    audio: np.ndarray,
    segments: List[Dict],
    sample_rate: int = 16000,
    hf_token: Optional[str] = None,
) -> Dict[str, Dict]:
    """Extract one embedding per speaker from diarized segments.

    For each speaker, all their speech segments are concatenated into a single
    waveform and a single embedding is computed from that.

    Args:
        audio: Full audio waveform (1-D numpy array, mono, float32).
        segments: List of dicts with 'start', 'end', 'speaker' keys.
        sample_rate: Sample rate of the audio.
        hf_token: HuggingFace token for gated model access.

    Returns:
        Dict mapping speaker labels to embedding info, e.g.:
        {
            "SPEAKER_00": {
                "embedding": [0.023, -0.156, ...],
                "embedding_dim": 512,
                "speech_duration": 45.2
            }
        }
    """
    from pyannote.audio import Inference

    model = _get_embedding_model(hf_token)
    inference = Inference(model, window="whole")

    # Group segments by speaker
    speaker_segments: Dict[str, List[Dict]] = {}
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker:
            speaker_segments.setdefault(speaker, []).append(seg)

    result = {}
    for speaker, segs in speaker_segments.items():
        chunks = []
        total_duration = 0.0
        for seg in segs:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            chunk = audio[start_sample:end_sample]
            if len(chunk) > 0:
                chunks.append(chunk)
                total_duration += seg["end"] - seg["start"]

        if not chunks:
            continue

        concatenated = np.concatenate(chunks)

        # Skip speakers with very little speech (< 1 second)
        if len(concatenated) < sample_rate:
            logger.warning(
                f"Skipping embedding for {speaker}: only {total_duration:.1f}s of speech"
            )
            continue

        waveform = torch.from_numpy(concatenated.copy()).unsqueeze(0).float()
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        embedding = inference(audio_input)

        result[speaker] = {
            "embedding": embedding.tolist(),
            "embedding_dim": len(embedding),
            "speech_duration": round(total_duration, 1),
        }

    return result


def validate_single_speaker(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hf_token: Optional[str] = None,
) -> tuple:
    """Check if audio contains a single speaker using diarization.

    Args:
        audio: 1-D numpy array of audio samples.
        sample_rate: Sample rate.
        hf_token: HuggingFace token.

    Returns:
        Tuple of (is_single_speaker: bool, speaker_count: int).
    """
    from pyannote.audio import Pipeline

    token = hf_token or os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=token
    )

    waveform = torch.from_numpy(audio.copy()).unsqueeze(0).float()
    input_data = {"waveform": waveform, "sample_rate": sample_rate}

    diarization = pipeline(input_data)

    speakers = set()
    for _, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)

    return len(speakers) <= 1, len(speakers)
