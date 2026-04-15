"""
Whisper-based automatic language detection.

Used to provide a `language=` hint to backends that don't do their own
language ID (Granite Speech, Voxtral when prompted, etc.). Whisper itself
already detects internally so this helper is bypassed for whisper:* backends.

Default probe model is whisper-tiny: language ID only uses the encoder + the
first decoder token, where tiny is ~as accurate as larger Whispers on >5s of
clean speech (~95% on major languages) but ~20x smaller, ~1s to load, and
doesn't compete for VRAM with the main transcription model.
"""

import logging
import threading
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Probe length: Whisper's mel front-end is fixed at 30s. Anything beyond is
# truncated by the processor anyway, so we cap explicitly to keep the GPU
# work bounded and avoid copying long arrays.
_PROBE_SECONDS = 30
_SAMPLE_RATE = 16000

# Languages supported by Whisper that we actually care about exposing. The
# raw Whisper vocab covers ~100; we let the model decide and only normalize
# the result to ISO-639-1 lowercase strings.


class WhisperLanguageDetector:
    """
    Lazy-loaded Whisper language detector.

    Thread-safe, single-instance per process. The model is loaded on first
    `detect()` call so importing this module is cheap.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-tiny",
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_id = model_id
        self._requested_device = device
        self._requested_dtype = torch_dtype

        self._model = None
        self._processor = None
        self._device: Optional[str] = None
        self._dtype: Optional[torch.dtype] = None
        self._load_lock = threading.Lock()

    # ── Lazy load ──────────────────────────────────────────────────────
    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            device = self._requested_device
            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

            if self._requested_dtype is not None:
                dtype = self._requested_dtype
            else:
                # bf16 on CUDA/Blackwell, fp32 on CPU. Tiny is small enough
                # that fp16 buys nothing measurable.
                dtype = torch.bfloat16 if "cuda" in device else torch.float32

            logger.info("Loading language detector: %s on %s (%s)", self.model_id, device, dtype)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            self._device = device
            self._dtype = dtype

    # ── Public API ─────────────────────────────────────────────────────
    def detect(self, audio: np.ndarray) -> Optional[str]:
        """
        Return an ISO-639-1 language code (e.g. "en", "fr") or None if
        detection failed. `audio` is mono float32 at 16 kHz.
        """
        if audio is None or len(audio) == 0:
            return None

        # Trim to the probe window. We slice from the start because most
        # recordings have informative speech early; no benefit from sampling
        # the middle for a fixed-length encoder.
        max_samples = _PROBE_SECONDS * _SAMPLE_RATE
        probe = audio[:max_samples]

        try:
            self._ensure_loaded()
        except Exception as e:
            logger.warning("Language detector failed to load (%s); skipping detection", e)
            return None

        try:
            input_features = self._processor(
                probe,
                sampling_rate=_SAMPLE_RATE,
                return_tensors="pt",
            ).input_features.to(self._device, dtype=self._dtype)

            with torch.inference_mode():
                # detect_language returns a tensor of language token IDs.
                # Available since transformers ≥4.39 on Whisper models.
                lang_token_ids = self._model.detect_language(input_features)

            # token id -> "<|fr|>" -> "fr"
            tokens = self._processor.tokenizer.convert_ids_to_tokens(
                lang_token_ids.flatten().tolist()
            )
            if not tokens:
                return None
            tok = tokens[0]
            if tok.startswith("<|") and tok.endswith("|>"):
                code = tok[2:-2].lower()
                # Whisper sometimes emits non-language special tokens if the
                # audio is silent/garbled. Filter to 2- or 3-letter codes.
                if 2 <= len(code) <= 3 and code.isalpha():
                    return code
            return None
        except AttributeError:
            # Older transformers without detect_language — fall back to a
            # manual single-step decode.
            return self._detect_via_first_token(probe)
        except Exception as e:
            logger.warning("Language detection failed (%s); proceeding without hint", e)
            return None

    # ── Fallback ───────────────────────────────────────────────────────
    def _detect_via_first_token(self, probe: np.ndarray) -> Optional[str]:
        """Manual language detection for transformers without detect_language()."""
        try:
            input_features = self._processor(
                probe, sampling_rate=_SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(self._device, dtype=self._dtype)

            decoder_start = torch.tensor(
                [[self._model.config.decoder_start_token_id]],
                device=self._device,
            )
            with torch.inference_mode():
                logits = self._model(
                    input_features=input_features,
                    decoder_input_ids=decoder_start,
                ).logits[:, -1]

            # Only consider language tokens
            tokenizer = self._processor.tokenizer
            lang_token_ids = []
            for tok, tid in tokenizer.get_added_vocab().items():
                if tok.startswith("<|") and tok.endswith("|>"):
                    code = tok[2:-2]
                    if 2 <= len(code) <= 3 and code.isalpha():
                        lang_token_ids.append((tid, code))

            if not lang_token_ids:
                return None
            ids = torch.tensor([t[0] for t in lang_token_ids], device=logits.device)
            scores = logits[0, ids]
            best = int(torch.argmax(scores).item())
            return lang_token_ids[best][1].lower()
        except Exception as e:
            logger.warning("Manual language detection failed (%s)", e)
            return None
