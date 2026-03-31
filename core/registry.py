import importlib
import os
from typing import Dict, Any, Optional

import yaml

from core.base import BaseTranscriber

# ── Config loading ──────────────────────────────────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.yaml")

def _load_models_config() -> Dict[str, Dict[str, Any]]:
    """Load model definitions from models.yaml."""
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(
            f"Model config not found at {_CONFIG_PATH}. "
            "Please create a models.yaml file (see README)."
        )
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def _get_models() -> Dict[str, Dict[str, Any]]:
    """Return the full model registry dict."""
    return _load_models_config()


# ── Public API ──────────────────────────────────────────────────

def normalize_model_spec(model_spec: str) -> str:
    """Normalize model specifier (strip, fuzzy prefix)."""
    model_spec = model_spec.strip()
    models = _get_models()
    
    if model_spec in models:
        return model_spec

    if ":" not in model_spec:
        # Fuzzy match: try common providers
        for provider in ["whisper", "moonshine", "voxtral", "canary"]:
            if f"{provider}:{model_spec}" in models:
                return f"{provider}:{model_spec}"
    
    return model_spec

def create_transcriber(
    model_spec: str,
    device: str = "auto",
    language: Optional[str] = None,
    **kwargs,
) -> BaseTranscriber:
    """
    Factory: create a transcriber instance from a model specifier.
    
    Looks up model_spec in models.yaml first, then falls back to
    treating it as a raw HuggingFace model ID for VoxtralTranscriber.
    """
    model_spec = normalize_model_spec(model_spec)
    models = _get_models()

    if model_spec in models:
        config = models[model_spec]
        module_path = config["module"]
        class_name = config["class"]
        default_args = dict(config.get("args", {}))

        # Merge caller kwargs (e.g. precision, flash_attn)
        default_args.update(kwargs)

        # Lazy import
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        return cls(device=device, language=language, **default_args)

    # Fallback: treat as a raw HuggingFace model path
    if "/" in model_spec or model_spec.startswith("./"):
        from core.transcribe import VoxtralTranscriber
        return VoxtralTranscriber(
            model_id=model_spec, device=device, language=language, **kwargs
        )

    raise ValueError(
        f"Unknown model specifier: {model_spec}. "
        f"Available: {', '.join(list_supported_models())}"
    )


def list_supported_models(enabled_only: bool = False) -> list[str]:
    """
    Return sorted list of model specifiers from models.yaml.
    
    If enabled_only=True, only return models with enabled: true.
    This is used by --model all.
    """
    models = _get_models()
    if enabled_only:
        return sorted(k for k, v in models.items() if v.get("enabled", True))
    return sorted(models.keys())
