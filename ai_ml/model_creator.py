"""
Singleton model loaders for Whisper and OpenAI.

Whisper is still loaded locally because STT remains on-device. The OpenAI
client is a lightweight singleton that centralizes API configuration and
is reused across the application.
"""

from __future__ import annotations

import logging

from openai import OpenAI

from app.config import settings
from ai_ml.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


# Optional torch (GPU detection for Whisper)
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _default_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Whisper

class WhisperModelLoader:
    """
    Lazy singleton loader for OpenAI Whisper.

    The model is downloaded/loaded only on the first call to
    :meth:`get_model`.  Subsequent calls return the cached instance.
    """

    _instance = None

    @classmethod
    def get_model(cls):
        """
        Return the cached Whisper model, loading it if necessary.

        Returns:
            whisper.Whisper: Loaded Whisper model instance.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        if cls._instance is None:
            try:
                import whisper  # noqa: PLC0415
                size = settings.WHISPER_MODEL_SIZE
                device = _default_device()
                logger.info("Loading Whisper '%s' model on %s …", size, device)
                cls._instance = whisper.load_model(size, device=device)
                logger.info("Whisper model loaded successfully.")
            except Exception as exc:
                raise ModelLoadError(f"Failed to load Whisper model: {exc}") from exc
        return cls._instance


# OpenAI

class OpenAIClientLoader:
    """
    Lazy singleton loader for the OpenAI client.

    The client is configured once from ``app.config.settings`` and reused
    across requests. It does not make a network call during startup, so
    validation failures remain cheap and explicit.
    """

    _instance: OpenAI | None = None

    @classmethod
    def get_client(cls) -> OpenAI:
        if cls._instance is None:
            try:
                if not settings.OPENAI_API_KEY:
                    raise ModelLoadError("OPENAI_API_KEY is not configured.")
                logger.info(
                    "Initialising OpenAI client with model defaults: question=%s, rubric=%s, eval=%s, mcq=%s",
                    settings.OPENAI_MODEL_QUESTION,
                    settings.OPENAI_MODEL_RUBRIC,
                    settings.OPENAI_MODEL_EVAL,
                    settings.OPENAI_MODEL_MCQ,
                )
                cls._instance = OpenAI(
                    api_key=settings.OPENAI_API_KEY or None,
                    organization=settings.OPENAI_ORGANIZATION_ID or None,
                    project=settings.OPENAI_PROJECT_ID or None,
                    base_url=settings.OPENAI_BASE_URL or None,
                    timeout=settings.OPENAI_TIMEOUT_SECONDS,
                    max_retries=settings.OPENAI_MAX_RETRIES,
                )
                logger.info("OpenAI client initialised successfully.")
            except Exception as exc:
                raise ModelLoadError(f"Failed to initialise OpenAI client: {exc}") from exc
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
