"""
Voxtral transcription via a remote vLLM OpenAI-compatible server.

This backend does NOT load the Voxtral model into the VoxHub process.
Instead, it forwards each segment to a vLLM HTTP server that already has
the model resident in GPU memory. Typical deployment: the ``voxtral-vllm``
service in ``docker-compose.yaml`` (activated with ``--profile vllm``).

Why a separate backend?
-----------------------
The transformers ``VoxtralTranscriber`` is convenient for development and
low-traffic setups, but vLLM ships a PagedAttention + continuous-batching
server that is typically 3–10x faster on Voxtral under load. Both coexist:
users pick which one by selecting the model key in ``models.yaml``
(``voxtral:mini-3b`` vs ``voxtral:mini-3b-vllm``).

Memory footprint
----------------
Zero on the VoxHub Python side: this class holds only an OpenAI HTTP client.
The actual Voxtral weights live in the vllm container. See the README for
details on enabling / disabling the vllm compose profile.

Endpoint
--------
We call ``POST {base_url}/audio/transcriptions`` — the OpenAI-compatible
transcription endpoint that vLLM mainline exposes for audio LLMs including
Voxtral. The realtime WebSocket (``/v1/realtime``) is NOT used here; that
path is reserved for future streaming support.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import wave
from typing import List, Optional

import numpy as np

from core.base import BaseTranscriber

logger = logging.getLogger(__name__)

# Default connection parameters. Overridable per-instance via constructor
# or globally via environment variables (useful for docker-compose).
_DEFAULT_BASE_URL = os.environ.get(
    "VOXTRAL_VLLM_URL", "http://voxtral-vllm:8000/v1"
)
_DEFAULT_TIMEOUT = float(os.environ.get("VOXTRAL_VLLM_TIMEOUT", "120"))
_DEFAULT_RETRIES = int(os.environ.get("VOXTRAL_VLLM_RETRIES", "3"))


def _ndarray_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Serialise a mono float32 np.ndarray to WAV-in-memory (16-bit PCM).

    vLLM's ``/v1/audio/transcriptions`` handler uses soundfile under the
    hood, so any of WAV/FLAC/OGG would work — we use WAV for zero-dep
    encoding and because it matches what the OpenAI client would upload.
    """
    if audio.dtype != np.int16:
        # Clip to [-1, 1] then scale — same recipe as the existing core/audio
        # loader uses on the decode side.
        clipped = np.clip(audio, -1.0, 1.0)
        audio_i16 = (clipped * 32767.0).astype(np.int16)
    else:
        audio_i16 = audio

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())
    return buf.getvalue()


class VoxtralVLLMTranscriber(BaseTranscriber):
    """Voxtral ASR via a remote vLLM server.

    Args mirror the transformers-based ``VoxtralTranscriber`` where
    possible, so swapping between backends in ``models.yaml`` requires no
    other code changes.

    Parameters
    ----------
    model_id:
        HuggingFace model id. Must match what the vLLM server was started
        with (``--model mistralai/Voxtral-Mini-3B-2507`` etc.).
    base_url:
        vLLM server root + ``/v1`` (OpenAI-compatible path prefix).
    api_key:
        Only required if you started vLLM with ``--api-key``. vLLM accepts
        any non-empty string otherwise, so we pass ``"EMPTY"`` by default.
    timeout, max_retries:
        HTTP tuning. Defaults are intentionally generous because the
        vllm container may take 30–60s to warm up on first request.
    sample_rate:
        Input sample rate of the ``audio`` arrays this transcriber will
        receive. VoxHub always pre-resamples to 16 kHz upstream, so the
        default rarely needs to change.
    """

    def __init__(
        self,
        model_id: str = "mistralai/Voxtral-Mini-3B-2507",
        device: str = "auto",  # kept for API compatibility; ignored
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_id=model_id, device=device, language=language)
        self.base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("VOXTRAL_VLLM_API_KEY", "EMPTY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.sample_rate = sample_rate

        # Voxtral supports context-carry semantically (via the `prompt` field
        # on the transcription endpoint). We keep it enabled so the
        # orchestration layer in api/transcriber.py passes context through.
        self.supports_context_carry = True

        # Lazy-created OpenAI client
        self._client = None

        # Silently drop unknown kwargs (flash_attn, compile_model, precision…)
        # so that the same model_kwargs payload the API service builds for
        # the transformers backend also works here.
        if kwargs:
            logger.debug(
                "VoxtralVLLMTranscriber: ignoring backend-specific kwargs %s",
                list(kwargs.keys()),
            )

    # ── Lifecycle ──────────────────────────────────────────────────────
    def load(self) -> None:
        """Initialise the HTTP client and probe the server.

        Unlike the transformers backend, ``load()`` here does NOT load any
        weights — the vLLM server already has the model in memory. We just
        verify it's reachable and that it's serving the expected model so
        we fail fast if the compose profile wasn't started.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "The vLLM backend requires the `openai` Python package. "
                "Add `openai>=1.0` to requirements.txt or install it manually."
            ) from e

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # Probe /models — this is cheap, does not run inference, and tells
        # us both (a) that the server is reachable and (b) which model id
        # it's actually serving. We only warn (not fail) on mismatch because
        # vLLM sometimes normalises the model id.
        try:
            models = self._client.models.list()
            served_ids = [m.id for m in getattr(models, "data", [])]
            if served_ids and self.model_id not in served_ids:
                logger.warning(
                    "vLLM server at %s serves %s but this backend is configured "
                    "for %s — requests will use '%s' as the model parameter and "
                    "the server may 404 or coerce.",
                    self.base_url, served_ids, self.model_id, self.model_id,
                )
            else:
                logger.info(
                    "vLLM backend ready: %s serving %s",
                    self.base_url, self.model_id,
                )
        except Exception as e:
            # Most likely: connection refused because the vllm service isn't
            # running. We raise a clearly-actionable message instead of letting
            # the first transcription request surface a cryptic error.
            raise RuntimeError(
                f"Cannot reach vLLM server at {self.base_url}: {e}. "
                f"Is the 'voxtral-vllm' service running? "
                f"Start it with: docker compose --profile vllm up -d voxtral-vllm"
            ) from e

    # ── Inference ──────────────────────────────────────────────────────
    def transcribe_segment(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """Transcribe one audio chunk via the remote vLLM server."""
        if self._client is None:
            self.load()

        wav_bytes = _ndarray_to_wav_bytes(audio, self.sample_rate)

        # OpenAI-style file tuple: (filename, bytes, content_type)
        # The filename is cosmetic; vLLM/soundfile sniffs the format from the
        # bytes, but a hint never hurts.
        file_tuple = ("segment.wav", wav_bytes, "audio/wav")

        # Build kwargs. ``prompt`` is the OpenAI-documented field for
        # "optional text to guide the model's style"; for Voxtral this maps
        # to the same "text prefix" role as context= in the transformers path.
        # We request "json" rather than "text" because vLLM's Voxtral
        # transcription handler (≥ 0.19) sometimes emits JSON-encoded
        # payloads with a plain Content-Type when "text" is requested,
        # which makes the OpenAI SDK hand us the raw JSON string verbatim.
        # The "json" path lets the SDK parse into a TranscriptionResponse
        # and _extract_text() below handles both the clean and the
        # degenerate (concatenated-SSE) shapes defensively.
        kwargs = {
            "file": file_tuple,
            "model": self.model_id,
            "response_format": "json",
        }
        # Only forward the language hint when we actually have one;
        # passing language="" or "auto" confuses some vLLM builds.
        if language and language.lower() != "auto":
            kwargs["language"] = language
        if context:
            # Trim: Voxtral's style-prompt window is small, and passing
            # multi-paragraph context risks exceeding it silently.
            kwargs["prompt"] = " ".join(context.split()[-30:])

        text = self._post_with_retries(kwargs)
        return text.strip()

    def transcribe_batch(self, audio_segments: List[np.ndarray]) -> List[str]:
        """Sequential for now — vLLM will still batch internally across
        concurrent requests, but true client-side batching on this endpoint
        is not part of the OpenAI spec. If you need maximum throughput,
        fire multiple requests concurrently upstream."""
        if not audio_segments:
            return []
        return [self.transcribe_segment(seg) for seg in audio_segments]

    # ── Internal helpers ───────────────────────────────────────────────
    def _post_with_retries(self, kwargs: dict) -> str:
        """POST the transcription request with simple exponential backoff.

        We retry on connection errors and 5xx-shaped exceptions. 4xx errors
        (bad model id, malformed audio) are re-raised immediately since
        they won't be fixed by retrying.
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.audio.transcriptions.create(
                    timeout=self.timeout, **kwargs
                )
                return self._extract_text(resp)
            except Exception as e:
                last_err = e
                # Fail fast on client-side errors (wrong model, auth…):
                status_code = getattr(e, "status_code", None)
                if status_code is not None and 400 <= status_code < 500:
                    raise
                if attempt >= self.max_retries:
                    break
                backoff = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "vLLM transcription failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, self.max_retries, e, backoff,
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"vLLM transcription failed after {self.max_retries} attempts: {last_err}"
        ) from last_err

    @staticmethod
    def _extract_text(resp) -> str:
        """Pull the plain transcription out of whatever shape vLLM returned.

        Normal case: the OpenAI SDK parses the JSON body into a
        ``TranscriptionResponse`` with a ``.text`` attribute.

        Degenerate case (observed on vLLM 0.19.x dev with Voxtral): the
        server streams N small JSON envelopes of the form
        ``{"text":"…","usage":{"type":"duration","seconds":N}}`` and the
        SDK returns the raw concatenated body as a Python ``str``. Older
        code then surfaced that JSON-as-string straight into the UI,
        producing lines like::

            {"text":"Hello","usage":{...}} {"text":"world","usage":{...}}

        We handle both: dict/object → ``.text``; string → try JSON-parse,
        walking multiple concatenated JSON values if present, and
        fall back to returning the trimmed string otherwise.
        """
        # OpenAI SDK object with .text
        text_attr = getattr(resp, "text", None)
        if isinstance(text_attr, str):
            return text_attr

        # Dict (if someone passed in the raw response.json())
        if isinstance(resp, dict) and isinstance(resp.get("text"), str):
            return resp["text"]

        # Plain string — either clean text or JSON-wrapped chunks
        if isinstance(resp, str):
            stripped = resp.strip()
            if not stripped:
                return ""
            if not (stripped.startswith("{") or stripped.startswith("[")):
                return stripped
            decoder = json.JSONDecoder()
            texts: List[str] = []
            idx = 0
            n = len(stripped)
            while idx < n:
                while idx < n and stripped[idx].isspace():
                    idx += 1
                if idx >= n:
                    break
                try:
                    obj, end = decoder.raw_decode(stripped, idx)
                except json.JSONDecodeError:
                    return " ".join(texts).strip() if texts else stripped
                idx = end
                if isinstance(obj, dict):
                    t = obj.get("text")
                    if isinstance(t, str) and t:
                        texts.append(t)
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict):
                            t = item.get("text")
                            if isinstance(t, str) and t:
                                texts.append(t)
            return " ".join(texts).strip() if texts else stripped

        return ""
