"""
IBM Granite Speech backend.

Granite Speech is a multimodal speech-language model: a speech encoder feeds a
text LLM, and transcription is expressed as a chat completion where the user
turn contains an `<|audio|>` placeholder followed by an instruction. The model
ships with a LoRA adapter that transformers enables/disables automatically,
so `peft` must be installed.

Supported languages: English, French, German, Spanish, Portuguese, Japanese.
Reference: https://huggingface.co/ibm-granite/granite-4.0-1b-speech
"""

import logging
import numpy as np
import torch
from typing import Optional, List

from transformers.utils import logging as hf_logging

from core.base import BaseTranscriber
from core.platform import detect_platform, get_optimal_device_map

# These are only available on transformers releases that ship the Granite
# Speech model. We import lazily so older envs still import the module.
try:
    from transformers import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor
    _HAS_GRANITE_SPEECH = True
except ImportError:
    GraniteSpeechForConditionalGeneration = None  # type: ignore
    GraniteSpeechProcessor = None  # type: ignore
    _HAS_GRANITE_SPEECH = False

logger = logging.getLogger(__name__)

# Default instruction given to the LLM after the audio token. IBM's own
# examples and the model card use this exact phrasing — sticking to it keeps
# the LoRA adapter in its trained distribution.
_DEFAULT_ASR_PROMPT = "can you transcribe the speech into a written format?"

# Per-language prompt variants. We keep English as the base and only override
# where a localized prompt materially helps — Granite is instruction-tuned
# and follows the English prompt across all supported languages, but a matched
# prompt tends to reduce code-switch artifacts on short utterances.
_LANG_PROMPTS = {
    "en": _DEFAULT_ASR_PROMPT,
    "fr": "peux-tu transcrire la parole au format écrit ?",
    "de": "kannst du die Sprache in Textform transkribieren?",
    "es": "¿puedes transcribir el habla a formato escrito?",
    "pt": "você pode transcrever a fala em formato escrito?",
    "ja": "音声を書き起こして文字にしてください。",
}


class GraniteSpeechTranscriber(BaseTranscriber):
    """
    Granite Speech backend using Hugging Face transformers.
    """

    def __init__(
        self,
        model_id: str = "ibm-granite/granite-4.0-1b-speech",
        device: str = "auto",
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_id, device)
        self.attn_implementation = attn_implementation
        self._dtype_override = torch_dtype  # optional string: "bfloat16", "float16", "auto"

        # Granite accepts a text prefix via the chat template — we use it to
        # feed prior-segment context, which helps short utterances
        # ("Okay", "Merci") be disambiguated as dialogue rather than isolated
        # tokens. See _build_chat() below.
        self.supports_context_carry = True

        self.model = None
        self.processor = None
        self.platform = None
        self._input_dtype = None

    # ── Load ───────────────────────────────────────────────────────────
    def load(self):
        if not _HAS_GRANITE_SPEECH:
            raise RuntimeError(
                "Granite Speech is not available in this transformers install. "
                "Upgrade transformers to a release that includes "
                "GraniteSpeechForConditionalGeneration (v5.0+)."
            )

        if self.device == "auto":
            self.platform = detect_platform()
        else:
            self.platform = {
                "cuda": detect_platform() if detect_platform() == "blackwell" else "cuda",
                "rocm": "rocm",
                "cpu": "cpu",
            }.get(self.device, detect_platform())

        device_map = get_optimal_device_map(self.platform)

        # Dtype: Granite Speech is distributed as bfloat16; that's also what
        # Blackwell's tensor cores prefer. Allow override via constructor.
        if self._dtype_override in ("bfloat16", "bf16"):
            torch_dtype = torch.bfloat16
        elif self._dtype_override in ("float16", "fp16"):
            torch_dtype = torch.float16
        elif self._dtype_override in ("auto", None):
            torch_dtype = torch.bfloat16 if self.platform in ("cuda", "blackwell") else torch.float32
        else:
            torch_dtype = torch.bfloat16

        self._input_dtype = torch_dtype

        # Attention backend: same guardrail we use for Voxtral — eager is the
        # safe default on Blackwell while SDPA kernels are still maturing on
        # sm_120. For Granite specifically SDPA usually works, but staying
        # conservative here avoids another cudaErrorNotPermitted surprise.
        if self.attn_implementation:
            attn_impl = self.attn_implementation
        elif self.platform == "blackwell":
            attn_impl = "eager"
        else:
            attn_impl = "sdpa"

        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self.processor = GraniteSpeechProcessor.from_pretrained(self.model_id)

            load_kwargs = dict(
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
            try:
                self.model = GraniteSpeechForConditionalGeneration.from_pretrained(
                    self.model_id, **load_kwargs
                )
            except (ValueError, TypeError) as e:
                if attn_impl != "eager" and "attn_implementation" in str(e).lower():
                    logger.warning(
                        "attn_implementation=%s rejected (%s); retrying with eager",
                        attn_impl, e,
                    )
                    load_kwargs["attn_implementation"] = "eager"
                    self.model = GraniteSpeechForConditionalGeneration.from_pretrained(
                        self.model_id, **load_kwargs
                    )
                else:
                    raise
        finally:
            hf_logging.set_verbosity(prev_verbosity)

    # ── Prompt construction ────────────────────────────────────────────
    def _asr_instruction(self, language: Optional[str]) -> str:
        if not language:
            return _DEFAULT_ASR_PROMPT
        return _LANG_PROMPTS.get(language.lower(), _DEFAULT_ASR_PROMPT)

    def _build_chat(self, language: Optional[str], context: Optional[str]):
        """
        Build the chat message list. The audio placeholder MUST appear in the
        user content — the processor expands it into audio embeddings.

        When `context` is provided (prior transcript), we stage it as a prior
        assistant turn so the LM continues the conversation rather than
        restarting. Keeping it as an assistant turn (not system) preserves
        the trained role distribution.
        """
        instruction = self._asr_instruction(language)
        messages = []
        if context:
            trimmed = " ".join(context.split()[-40:])
            messages.append({"role": "assistant", "content": trimmed})
        messages.append(
            {"role": "user", "content": f"<|audio|>{instruction}"}
        )
        return messages

    # ── Inference ──────────────────────────────────────────────────────
    def transcribe_segment(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("GraniteSpeechTranscriber: call load() before transcribe_segment().")

        chat = self._build_chat(language, context)
        prompt = self.processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Granite's processor accepts audio either as a numpy array or a torch
        # tensor. We standardize on a 1-D float32 torch tensor (mono, 16 kHz)
        # and let the processor handle feature extraction.
        audio_t = torch.as_tensor(audio, dtype=torch.float32).contiguous()
        if audio_t.ndim > 1:
            audio_t = audio_t.mean(dim=-1)  # downmix to mono if needed

        inputs = self.processor(
            prompt,
            audio_t,
            device=self.model.device,
            return_tensors="pt",
        )
        # Cast float tensors (audio features) to the model dtype. Integer
        # tensors (input_ids, attention_mask) stay untouched.
        target_dtype = getattr(self.model, "dtype", self._input_dtype)
        inputs = {
            k: (v.to(self.model.device, dtype=target_dtype) if v.is_floating_point()
                else v.to(self.model.device))
            for k, v in inputs.items()
        }

        max_new_tokens = self.estimate_max_tokens(audio)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        text = self.processor.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0]
        return text.strip()

    def transcribe_batch(self, audio_segments: List[np.ndarray]) -> List[str]:
        # Granite's chat-template path doesn't benefit much from batching
        # given variable prompt lengths; stick to the sequential default,
        # which is also what the official examples do.
        return super().transcribe_batch(audio_segments)
