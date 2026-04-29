"""
Application-level state container.

All models that are loaded once at startup and shared across requests
live here.  Import ``app_state`` wherever you need access to a preloaded
model rather than reaching into the module directly.

Usage::

    from app.core.state import app_state

    whisper = app_state.whisper_model   # may be None before startup completes
    ollama  = app_state.ollama_model    # may be None before startup completes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AppState:
    """Container for singleton model instances loaded at startup."""

    whisper_model: Optional[Any] = field(default=None)
    """OpenAI Whisper model instance."""

    ollama_model: Optional[Any] = field(default=None)
    """LangChain-wrapped Ollama (mistral:7b) model instance."""

    st_model: Optional[Any] = field(default=None)
    """SentenceTransformer model instance (for MCQ evaluation)."""

    @property
    def is_ready(self) -> bool:
        """Return ``True`` when all three core models are loaded."""
        return all([
            self.whisper_model is not None,
            self.ollama_model is not None,
            self.st_model is not None,
        ])

    @property
    def llm_ready(self) -> bool:
        """Return ``True`` when the Ollama model is loaded."""
        return self.ollama_model is not None

    @property
    def stt_ready(self) -> bool:
        """Return ``True`` when the Whisper model is loaded."""
        return self.whisper_model is not None

    @property
    def mcq_ready(self) -> bool:
        """Return ``True`` when the SentenceTransformer model is loaded."""
        return self.st_model is not None


# Single global instance — import this everywhere
app_state = AppState()
