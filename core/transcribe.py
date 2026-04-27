import torch
import logging
import numpy as np
from typing import Optional, List
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.utils import logging as hf_logging

# Voxtral has its own model class in transformers; prefer it over the generic
# AutoModelForSpeechSeq2Seq head, which doesn't expose Voxtral-specific kwargs.
try:
    from transformers import VoxtralRealtimeForConditionalGeneration
    _HAS_VOXTRAL_REALTIME = True
except ImportError:  # older transformers
    VoxtralRealtimeForConditionalGeneration = None  # type: ignore
    _HAS_VOXTRAL_REALTIME = False

try:
    from transformers import VoxtralForConditionalGeneration
    _HAS_VOXTRAL = True
except ImportError:
    VoxtralForConditionalGeneration = None  # type: ignore
    _HAS_VOXTRAL = False

from core.base import BaseTranscriber
from core.platform import (
    detect_platform,
    get_optimal_device_map,
    get_torch_dtype,
    supports_nvfp4,
    platform_summary,
)

# Suppress noisy warnings
logging.getLogger("accelerate.big_modeling").setLevel(logging.ERROR)

def _is_prequantized(model_id: str) -> str | None:
    if not model_id:
        return None
    model_lower = model_id.lower()
    for tag in ("fp8", "-fp8-", "_fp8_", "fp8-dynamic"):
        if tag in model_lower:
            return "fp8"
    for tag in ("gptq", "awq", "gguf"):
        if tag in model_lower:
            return tag
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        qc = getattr(config, "quantization_config", None)
        if qc:
            quant_method = qc.get("quant_method", None) if isinstance(qc, dict) else getattr(qc, "quant_method", None)
            if quant_method:
                return quant_method
    except Exception:
        pass
    return None

def _build_quantization_config(platform: str, precision: str, model_id: str = ""):
    # On high-end NVIDIA GPUs (Ampere/Blackwell), always use bfloat16 for compute
    compute_dtype = torch.bfloat16 if platform in ("cuda", "blackwell") else torch.float16

    if precision == "auto":
        prequant = _is_prequantized(model_id)
        if prequant:
            return None
        return _build_quantization_config(platform, "q4", model_id)

    if precision == "fp8":
        try:
            from transformers import TorchAoConfig
            return TorchAoConfig("float8_weight_only")
        except ImportError:
            return None

    if precision == "q8":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(load_in_8bit=True)

    if precision == "q4":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    return None

class VoxtralTranscriber(BaseTranscriber):
    def __init__(
        self,
        model_id: str = "mistralai/Voxtral-Mini-3B-2507",
        device: str = "auto",
        precision: str = "auto",
        flash_attn: bool = False,
        compile_model: bool = False,
        transcription_delay_ms: int = 480,
        attn_implementation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_id, device)
        self.precision = precision
        self.flash_attn = flash_attn
        self.compile_model = compile_model
        self.transcription_delay_ms = transcription_delay_ms
        # If unset, we choose at load() time based on the platform. On Blackwell
        # (GB10) the SDPA sliding-window attention kernel currently crashes
        # with cudaErrorNotPermitted on Voxtral, so we default to "eager" there.
        self.attn_implementation = attn_implementation
        self.supports_context_carry = True

        # Internal state
        self.model = None
        self.processor = None
        self.platform = None
        self._input_dtype = None
        self._is_realtime = False

    def load(self):
        if self.device == "auto":
            self.platform = detect_platform()
        else:
            self.platform = {
                "cuda": detect_platform() if detect_platform() == "blackwell" else "cuda",
                "rocm": "rocm",
                "cpu": "cpu",
            }.get(self.device, detect_platform())

        device_map = get_optimal_device_map(self.platform)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        
        self._is_realtime = hasattr(self.processor, 'feature_extractor') and \
            type(self.processor).__name__ == "VoxtralRealtimeProcessor"

        if self._is_realtime and self.transcription_delay_ms != 480:
            self._set_transcription_delay(self.transcription_delay_ms)

        quantization_config = _build_quantization_config(self.platform, self.precision, self.model_id)
        torch_dtype = get_torch_dtype(self.platform, self.precision)

        # Resolve attention implementation:
        #   explicit constructor arg > flash_attn flag > platform default
        #
        # Platform defaults (as of 2026-04):
        #   - blackwell (sm_120/sm_121, GB10/B200 consumer): "eager"
        #       * SDPA sliding-window kernel still raises cudaErrorNotPermitted
        #         on Voxtral in current torch builds.
        #       * Upstream Dao-AILab/flash-attention does not ship official
        #         wheels for sm_120+; FA4 targets B200/GB200 only. Community
        #         wheels exist but we don't want to force that choice.
        #   - cuda (sm_80/sm_86/sm_89/sm_90 Ampere/Ada/Hopper): "flash_attention_2"
        #       * Best throughput on Voxtral when the flash-attn wheel is
        #         available. If it's not installed (ImportError / ValueError
        #         at load time), the except-clause below falls back to eager
        #         and logs a clear hint.
        #   - rocm / cpu: "sdpa" (FA2 is NVIDIA-only).
        if self.attn_implementation:
            attn_impl = self.attn_implementation
        elif self.flash_attn:
            attn_impl = "flash_attention_2"
        elif self.platform == "blackwell":
            attn_impl = "eager"
        elif self.platform == "cuda":
            attn_impl = "flash_attention_2"
        else:
            # rocm, cpu, unknown — SDPA is safe everywhere
            attn_impl = "sdpa"

        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            load_kwargs = dict(
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config

            # Pick the most specific model class available, per the official
            # Voxtral docs. Falls back to AutoModelForSpeechSeq2Seq for older
            # transformers releases.
            model_cls = None
            if self._is_realtime and _HAS_VOXTRAL_REALTIME:
                model_cls = VoxtralRealtimeForConditionalGeneration
            elif (not self._is_realtime) and _HAS_VOXTRAL:
                model_cls = VoxtralForConditionalGeneration
            else:
                model_cls = AutoModelForSpeechSeq2Seq

            try:
                self.model = model_cls.from_pretrained(self.model_id, **load_kwargs)
            except (ValueError, TypeError, ImportError) as e:
                # Attention backend not supported — retry with eager.
                # ImportError catches the "flash-attn is not installed" case
                # that transformers raises when attn_implementation=flash_attention_2
                # is requested without the flash-attn wheel available.
                err_text = str(e).lower()
                is_attn_error = (
                    "attn_implementation" in err_text
                    or "flash_attn" in err_text
                    or "flash-attn" in err_text
                )
                if attn_impl != "eager" and is_attn_error:
                    log = logging.getLogger(__name__)
                    if attn_impl == "flash_attention_2":
                        log.warning(
                            "Flash Attention 2 unavailable (%s); falling back to 'eager'. "
                            "Install `flash-attn` in your CUDA image for ~1.5-2x faster "
                            "Voxtral inference. See requirements-cuda.txt.",
                            e,
                        )
                    else:
                        log.warning(
                            "attn_implementation=%s rejected (%s); retrying with 'eager'",
                            attn_impl, e,
                        )
                    load_kwargs["attn_implementation"] = "eager"
                    self.model = model_cls.from_pretrained(self.model_id, **load_kwargs)
                else:
                    raise

            if self.compile_model:
                self.model = torch.compile(self.model, mode="reduce-overhead")
        finally:
            hf_logging.set_verbosity(prev_verbosity)

        # Ensure input dtype matches internal compute/residual dtype
        # On Ampere/Blackwell, we almost always want BFloat16 to avoid Conv1d bias crashes
        if self.platform in ("cuda", "blackwell"):
            self._input_dtype = torch.bfloat16
        else:
            self._input_dtype = torch.float16

    def _set_transcription_delay(self, delay_ms: int):
        valid_values = list(range(80, 1280, 80)) + [2400]
        if delay_ms not in valid_values:
            raise ValueError(f"Invalid delay {delay_ms}ms.")
        tokenizer = self.processor.tokenizer
        tekken = getattr(tokenizer, '_tekken', None)
        if tekken and hasattr(tekken, 'audio_encoder'):
            tekken.audio_encoder.audio_config.transcription_delay_ms = delay_ms

    def transcribe_segment(
        self, 
        audio: np.ndarray, 
        language: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        inputs = self._prepare_inputs(audio, context, language=language)
        with torch.inference_mode():
            max_new_tokens = self.estimate_max_tokens(audio)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        if self._is_realtime:
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            prompt_len = inputs.input_ids.shape[1]
            transcription = self.processor.batch_decode(
                generated_ids[:, prompt_len:], skip_special_tokens=True
            )[0]
        return transcription.strip()

    def transcribe_batch(self, audio_segments: List[np.ndarray]) -> List[str]:
        if not audio_segments:
            return []
        if not self._is_realtime:
            return super().transcribe_batch(audio_segments)

        target_dtype = getattr(self.model, "dtype", self._input_dtype)
        inputs = self.processor(
            audio_segments,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device, dtype=target_dtype)

        longest = max(len(a) for a in audio_segments)
        max_new_tokens = max(64, int((longest / 16000) * 10))

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.strip() for t in transcriptions]

    def _prepare_inputs(self, audio: np.ndarray, context: Optional[str] = None, language: Optional[str] = None):
        # Always use the model's actual dtype to avoid bf16/fp16 mismatches
        # after quantization or auto-dtype loading.
        target_dtype = getattr(self.model, "dtype", self._input_dtype)

        if self._is_realtime:
            # Match the official Voxtral example signature exactly:
            #   processor(audio_array, return_tensors="pt")
            # Passing context as `text=` gives the LM a prior-turn prefix,
            # which materially improves short-utterance disambiguation
            # ("Okay" / "Merci" / "bien sûr" cases).
            kwargs = {"return_tensors": "pt"}
            if context:
                kwargs["text"] = " ".join(context.split()[-30:])
            inputs = self.processor(audio, **kwargs)
            return inputs.to(self.model.device, dtype=target_dtype)
        else:
            inputs = self.processor.apply_transcription_request(
                audio=[audio],
                format=["wav"],
                sampling_rate=16000,
                model_id=self.model_id,
            )
            return inputs.to(self.model.device, dtype=target_dtype)
