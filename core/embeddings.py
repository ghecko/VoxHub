"""
Speaker embedding extraction.

One embedding backend, chosen by ``VOXHUB_EMBEDDING_MODEL``, behind
:class:`EmbeddingBackend`. Every model family pyannote.audio can wrap is
accepted (``pyannote/embedding``, ``pyannote/wespeaker-voxceleb-resnet34-LM``,
``speechbrain/spkrec-ecapa-voxceleb``, NeMo TitaNet, ...), the vectors are
L2-normalised so cosine distance is the metric everywhere.

The embedding space is part of the API contract: VoxHub reports the model id
and dimension next to every embedding it returns (``speaker_embedding_model``),
and consumers (OpenHiNotes) refuse to compare vectors from another space.
Changing the model therefore invalidates every stored profile.

Why not stay on ``pyannote/embedding`` (2020, 512-d): measured on HiDock
meetings it puts the same person recorded on two microphones 0.36-0.58 apart
and a Microsoft Teams synthetic voice 0.22 from a real speaker, so no
threshold separates identities across sessions. See bench/embedding_models.py
to compare candidates on your own recordings before switching.

IMPORTANT: VoxHub never persists embeddings. They are computed on-the-fly
and returned in API responses. The consuming application is responsible
for storage and matching.
"""

import logging
import os
import threading
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default space, kept for backward compatibility with existing profiles.
DEFAULT_EMBEDDING_MODEL = "pyannote/embedding"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EmbeddingBackend:
    """A loaded speaker-embedding model with a uniform ``embed`` call."""

    def __init__(self, model_id: str, hf_token: Optional[str] = None, device: Optional[torch.device] = None):
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

        token = hf_token or os.getenv("HF_TOKEN")
        dev = device or _device()
        logger.info("Loading speaker embedding model: %s on %s", model_id, dev)
        try:
            self.model = PretrainedSpeakerEmbedding(model_id, device=dev, token=token)
        except TypeError:  # pyannote.audio < 3.3 spells it use_auth_token
            self.model = PretrainedSpeakerEmbedding(model_id, device=dev, use_auth_token=token)
        self.model_id = model_id
        self.dim = int(self.model.dimension)
        self.sample_rate = int(getattr(self.model, "sample_rate", 16000) or 16000)
        logger.info("Speaker embedding model loaded: %s (dim=%d)", model_id, self.dim)

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """L2-normalised embedding of a mono float32 waveform."""
        if sample_rate != self.sample_rate:
            import torchaudio.functional as F  # noqa: N812
            wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
            audio = F.resample(wav, sample_rate, self.sample_rate).numpy()
        waveforms = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).reshape(1, 1, -1)
        with torch.inference_mode():
            emb = self.model(waveforms)
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def info(self) -> Dict[str, object]:
        return {"id": self.model_id, "dim": self.dim}


_backend: Optional[EmbeddingBackend] = None
_backend_lock = threading.Lock()


def get_embedding_backend(model_id: Optional[str] = None, hf_token: Optional[str] = None) -> EmbeddingBackend:
    """Lazy singleton. ``model_id`` defaults to ``VOXHUB_EMBEDDING_MODEL``.

    One model per process: asking for a different id after the first load
    replaces it (the bench does that; the server never does).
    """
    global _backend
    wanted = model_id or os.getenv("VOXHUB_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    with _backend_lock:
        if _backend is None or _backend.model_id != wanted:
            _backend = EmbeddingBackend(wanted, hf_token)
        return _backend


def embedding_model_info(model_id: Optional[str] = None) -> Dict[str, object]:
    """``{"id", "dim"}`` of the active space (loads the model if needed)."""
    return get_embedding_backend(model_id).info()


def extract_embedding_from_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hf_token: Optional[str] = None,
    model_id: Optional[str] = None,
) -> List[float]:
    """Extract a single speaker embedding from a mono float32 waveform."""
    return get_embedding_backend(model_id, hf_token).embed(audio, sample_rate).tolist()


def _concat_speaker_audio(
    audio: np.ndarray, segs: List[Dict], sample_rate: int, max_seconds: Optional[float]
) -> tuple:
    """Concatenate a speaker's segments (longest first when capped)."""
    ordered = sorted(segs, key=lambda x: x["start"] - x["end"]) if max_seconds else segs
    chunks, total = [], 0.0
    for seg in ordered:
        if max_seconds and total >= max_seconds:
            break
        a, b = int(seg["start"] * sample_rate), int(seg["end"] * sample_rate)
        chunk = audio[a:b]
        if len(chunk):
            chunks.append(chunk)
            total += len(chunk) / sample_rate
    return (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)), total


def extract_per_speaker_embeddings(
    audio: np.ndarray,
    segments: List[Dict],
    sample_rate: int = 16000,
    hf_token: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Dict]:
    """One embedding per speaker from diarized segments (all of their speech).

    Returns ``{"SPEAKER_00": {"embedding": [...], "embedding_dim": N,
    "speech_duration": 45.2}, ...}``; speakers with < 1 s of speech are skipped.
    """
    backend = get_embedding_backend(model_id, hf_token)
    by_speaker: Dict[str, List[Dict]] = {}
    for seg in segments:
        if seg.get("speaker"):
            by_speaker.setdefault(seg["speaker"], []).append(seg)

    result = {}
    for speaker, segs in by_speaker.items():
        wav, total = _concat_speaker_audio(audio, segs, sample_rate, None)
        if len(wav) < sample_rate:
            logger.warning("Skipping embedding for %s: only %.1fs of speech", speaker, total)
            continue
        vec = backend.embed(wav, sample_rate)
        result[speaker] = {
            "embedding": vec.tolist(),
            "embedding_dim": len(vec),
            "speech_duration": round(total, 1),
        }
    return result


def cluster_embeddings(
    audio: np.ndarray,
    turns: List[Dict],
    sample_rate: int = 16000,
    hf_token: Optional[str] = None,
    max_seconds: float = 60.0,
    min_seconds: float = 1.0,
    model_id: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """One L2-normalised embedding per speaker label in ``turns``, for the
    cluster merge: audio capped at ``max_seconds`` per speaker (longest turns
    first) so a one-hour meeting does not push an hour of waveform through
    the model; speakers under ``min_seconds`` are skipped."""
    by_speaker: Dict[str, List[Dict]] = {}
    for t in turns:
        if t.get("speaker") and t["end"] > t["start"]:
            by_speaker.setdefault(t["speaker"], []).append(t)
    if not by_speaker:
        return {}
    backend = get_embedding_backend(model_id, hf_token)
    out: Dict[str, np.ndarray] = {}
    for speaker, segs in by_speaker.items():
        wav, total = _concat_speaker_audio(audio, segs, sample_rate, max_seconds)
        if total < min_seconds:
            continue
        out[speaker] = backend.embed(wav, sample_rate)
    return out


def validate_single_speaker(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hf_token: Optional[str] = None,
):
    """Check if audio contains a single speaker using diarization.

    Returns (is_single_speaker, speaker_count). Loads its own copy of the
    diarization pipeline; used only by the enrolment endpoint.
    """
    from pyannote.audio import Pipeline

    token = hf_token or os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        os.getenv("VOXHUB_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"), token=token
    )

    waveform = torch.from_numpy(audio.copy()).unsqueeze(0).float()
    input_data = {"waveform": waveform, "sample_rate": sample_rate}

    output = pipeline(input_data)
    diarization = output.speaker_diarization if hasattr(output, "speaker_diarization") else output

    speakers = set()
    for _, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)

    return len(speakers) <= 1, len(speakers)
