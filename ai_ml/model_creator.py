"""
Singleton model loaders for Whisper and Ollama (mistral:7b).

All heavy models are loaded **once** at application startup (via the
FastAPI lifespan) and reused for every request.  Each class also supports
lazy loading as a fallback so unit tests and scripts can import without
starting the full server.

Ollama must be running locally and the target model must be pulled:

    ollama pull mistral:7b

The Ollama base URL and model name are read from ``app.config.settings``.
"""

from __future__ import annotations

import logging

import httpx
from langchain_ollama import ChatOllama

from app.config import settings
from ai_ml.exceptions import ModelLoadError, OllamaConnectionError

logger = logging.getLogger(__name__)


# ── Optional torch (GPU detection for Whisper) ───────────────────────────────
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


# ── Ollama health / model probe ───────────────────────────────────────────────

def check_ollama_server(base_url: str | None = None) -> bool:
    """
    Verify that the Ollama server is reachable.

    Args:
        base_url: Override for the Ollama base URL. Defaults to
                  ``settings.OLLAMA_BASE_URL``.

    Returns:
        ``True`` if the server responded with HTTP 200.

    Raises:
        OllamaConnectionError: If the server is unreachable or returns an
                               unexpected response.
    """
    url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
    try:
        response = httpx.get(f"{url}/api/tags", timeout=5.0)
        response.raise_for_status()
        return True
    except httpx.ConnectError as exc:
        raise OllamaConnectionError(
            f"Cannot connect to Ollama at '{url}'. "
            "Make sure Ollama is running:  ollama serve"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaConnectionError(
            f"Ollama server at '{url}' returned HTTP {exc.response.status_code}."
        ) from exc
    except Exception as exc:
        raise OllamaConnectionError(
            f"Unexpected error while probing Ollama server at '{url}': {exc}"
        ) from exc


def check_ollama_model(model_name: str | None = None, base_url: str | None = None) -> bool:
    """
    Verify that the required Ollama model is available locally.

    Args:
        model_name: Override for the model name. Defaults to
                    ``settings.OLLAMA_MODEL_NAME``.
        base_url:   Override for the Ollama base URL. Defaults to
                    ``settings.OLLAMA_BASE_URL``.

    Returns:
        ``True`` if the model is in the local Ollama model list.

    Raises:
        OllamaConnectionError: If the server cannot be reached.
        ModelLoadError: If the specified model is not available locally.
    """
    url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
    name = model_name or settings.OLLAMA_MODEL_NAME

    try:
        response = httpx.get(f"{url}/api/tags", timeout=5.0)
        response.raise_for_status()
        data = response.json()
    except OllamaConnectionError:
        raise
    except Exception as exc:
        raise OllamaConnectionError(
            f"Could not retrieve model list from Ollama at '{url}': {exc}"
        ) from exc

    available = [m.get("name", "") for m in data.get("models", [])]
    # Normalize: "mistral:7b" matches "mistral:7b", "mistral:latest", etc.
    name_base = name.split(":")[0].lower()
    for m in available:
        if m.lower().startswith(name_base):
            logger.debug("Ollama model '%s' found in local model list.", m)
            return True

    pull_cmd = f"ollama pull {name}"
    raise ModelLoadError(
        f"Ollama model '{name}' is not available locally.\n"
        f"  Available models: {available or ['(none)']}\n"
        f"  Pull the model first:  {pull_cmd}"
    )


# ── Whisper ───────────────────────────────────────────────────────────────────

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


# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaModelLoader:
    """
    Lazy singleton loader for Ollama (mistral:7b) via LangChain.

    The :class:`~langchain_ollama.ChatOllama` wrapper is created only on
    the first call to :meth:`get_model` after verifying that:

    1. The Ollama server is reachable.
    2. The target model is available locally (otherwise a helpful error
       with the ``ollama pull`` command is raised).

    Configuration is read from ``app.config.settings``:

    - ``OLLAMA_BASE_URL``   — default ``http://localhost:11434``
    - ``OLLAMA_MODEL_NAME`` — default ``mistral:7b``
    - ``OLLAMA_TEMPERATURE``
    - ``OLLAMA_MAX_TOKENS``
    """

    _instance: ChatOllama | None = None

    @classmethod
    def get_model(cls) -> ChatOllama:
        """
        Return the cached Ollama model wrapper, creating it if necessary.

        Returns:
            ChatOllama: LangChain-wrapped Ollama model.

        Raises:
            OllamaConnectionError: If the Ollama server is unreachable.
            ModelLoadError:        If the model is not pulled locally.
        """
        if cls._instance is None:
            logger.info(
                "Connecting to Ollama at '%s' with model '%s' …",
                settings.OLLAMA_BASE_URL,
                settings.OLLAMA_MODEL_NAME,
            )

            # Pre-flight checks — fast-fail with actionable messages
            check_ollama_server(settings.OLLAMA_BASE_URL)
            check_ollama_model(settings.OLLAMA_MODEL_NAME, settings.OLLAMA_BASE_URL)

            try:
                cls._instance = ChatOllama(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_MODEL_NAME,
                    temperature=settings.OLLAMA_TEMPERATURE,
                    num_predict=settings.OLLAMA_MAX_TOKENS,
                )
                logger.info(
                    "Ollama model '%s' initialised successfully.",
                    settings.OLLAMA_MODEL_NAME,
                )
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to initialise Ollama model '{settings.OLLAMA_MODEL_NAME}': {exc}"
                ) from exc
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Clear the cached instance.

        Useful in tests or when the Ollama server is restarted and you need
        to force re-initialisation on the next :meth:`get_model` call.
        """
        cls._instance = None
