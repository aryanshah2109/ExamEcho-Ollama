"""
Singleton model loaders for Whisper and Groq (llama-3.3-70b-versatile).

All heavy models are loaded **once** at application startup (via the
FastAPI lifespan) and reused for every request.  Each class also supports
lazy loading as a fallback so unit tests and scripts can import without
starting the full server.

Groq API key and model name are read from ``app.config.settings``.
"""

from __future__ import annotations

import logging

from langchain_groq import ChatGroq

from app.config import settings
from ai_ml.exceptions import ModelLoadError, GroqConnectionError

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


# Groq

class GroqModelLoader:
    """
    Lazy singleton loader for Groq via LangChain.

    The :class:`~langchain_groq.ChatGroq` wrapper is created only on
    the first call to :meth:`get_model`.

    Configuration is read from ``app.config.settings``:

    - ``GROQ_API_KEY``     — required
    - ``GROQ_MODEL_NAME``  — default ``llama-3.3-70b-versatile``
    - ``GROQ_TEMPERATURE``
    - ``GROQ_MAX_TOKENS``
    """

    _instance: ChatGroq | None = None

    @classmethod
    def get_model(cls) -> ChatGroq:
        """
        Return the cached Groq model wrapper, creating it if necessary.

        Returns:
            ChatGroq: LangChain-wrapped Groq model.

        Raises:
            GroqConnectionError: If the Groq API key is missing or invalid.
            ModelLoadError:      If the model cannot be initialised.
        """
        if cls._instance is None:
            if not settings.GROQ_API_KEY:
                raise GroqConnectionError(
                    "GROQ_API_KEY is not set. "
                    "Add it to your .env file: GROQ_API_KEY=gsk_..."
                )

            logger.info(
                "Connecting to Groq with model '%s' …",
                settings.GROQ_MODEL_NAME,
            )

            try:
                cls._instance = ChatGroq(
                    api_key=settings.GROQ_API_KEY,
                    model=settings.GROQ_MODEL_NAME,
                    temperature=settings.GROQ_TEMPERATURE,
                    max_tokens=settings.GROQ_MAX_TOKENS,
                )
                logger.info(
                    "Groq model '%s' initialised successfully.",
                    settings.GROQ_MODEL_NAME,
                )
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to initialise Groq model '{settings.GROQ_MODEL_NAME}': {exc}"
                ) from exc
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Clear the cached instance.

        Useful in tests or when you need to force re-initialisation
        on the next :meth:`get_model` call.
        """
        cls._instance = None
