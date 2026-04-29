"""
Text-to-Speech module for ExamEcho.

Provides an abstract pipeline so the underlying TTS engine (currently gTTS)
can be swapped out without touching the service layer.

Typical usage::

    from ai_ml.tts import TTSPipeline, TTSConfig, DirectTextSource, GTTSEngine

    config = TTSConfig(language="en", return_bytes=True)
    pipeline = TTSPipeline(
        source=DirectTextSource("What is polymorphism?"),
        engine=GTTSEngine(),
        config=config,
    )
    audio_bytes = pipeline.run()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from gtts import gTTS

from ai_ml.exceptions import EngineError, TextSourceError, TTSError

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TTSConfig:
    """
    Parameters controlling TTS synthesis behaviour.

    Attributes:
        language:     BCP-47 language code (e.g. ``"en"``, ``"hi"``).
        slow:         Whether to generate slow-speed speech.
        output_file:  Path to write the audio file to (required unless
                      ``return_bytes=True``).
        return_bytes: If ``True``, return raw MP3 bytes instead of writing a file.
    """

    language: str = "en"
    slow: bool = False
    output_file: Optional[Path] = None
    return_bytes: bool = False


# ── Text sources ──────────────────────────────────────────────────────────────

class TextSource(ABC):
    """Abstract provider for the text to be synthesised."""

    @abstractmethod
    def get_text(self) -> str: ...


class DirectTextSource(TextSource):
    """Wraps an in-memory string."""

    def __init__(self, text: str) -> None:
        self._text = text.strip()

    def get_text(self) -> str:
        if not self._text:
            raise TextSourceError("Provided text is empty.")
        return self._text


class FileTextSource(TextSource):
    """Reads text from a UTF-8 encoded file."""

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()

    def get_text(self) -> str:
        if not self._path.exists():
            raise TextSourceError(f"Text file not found: {self._path}")
        try:
            text = self._path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise TextSourceError(f"Could not read text file: {exc}") from exc
        if not text:
            raise TextSourceError(f"Text file is empty: {self._path}")
        return text


# ── Engines ───────────────────────────────────────────────────────────────────

class TTSEngine(ABC):
    """Abstract TTS synthesis backend."""

    @abstractmethod
    def synthesize(self, text: str, config: TTSConfig) -> Union[bytes, Path]: ...


class GTTSEngine(TTSEngine):
    """
    TTS engine backed by Google Text-to-Speech (gTTS).

    Returns either raw MP3 bytes (when ``config.return_bytes=True``) or
    saves the audio to ``config.output_file`` and returns its path.
    """

    def synthesize(self, text: str, config: TTSConfig) -> Union[bytes, Path]:
        if not text:
            raise EngineError("Cannot synthesise empty text.")

        try:
            tts = gTTS(text=text, lang=config.language, slow=config.slow)
        except Exception as exc:
            raise EngineError(f"gTTS initialisation failed: {exc}") from exc

        if config.return_bytes:
            buf = BytesIO()
            try:
                tts.write_to_fp(buf)
            except Exception as exc:
                raise EngineError(f"gTTS write to buffer failed: {exc}") from exc
            logger.debug("TTS: generated %d bytes for '%s…'", buf.tell(), text[:40])
            return buf.getvalue()

        # ── file output path ──────────────────────────────────────────────────
        if not config.output_file:
            raise EngineError("output_file must be set when return_bytes=False.")

        output = config.output_file.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            tts.save(str(output))
        except Exception as exc:
            raise EngineError(f"gTTS save failed: {exc}") from exc

        logger.debug("TTS: audio saved to %s", output)
        return output


# ── Pipeline ──────────────────────────────────────────────────────────────────

class TTSPipeline:
    """
    Orchestrates text retrieval → synthesis.

    Args:
        source: A :class:`TextSource` that provides the input text.
        engine: A :class:`TTSEngine` that performs synthesis.
        config: :class:`TTSConfig` controlling synthesis behaviour.
    """

    def __init__(self, source: TextSource, engine: TTSEngine, config: TTSConfig) -> None:
        self.source = source
        self.engine = engine
        self.config = config

    def run(self) -> Union[bytes, Path]:
        """
        Execute the pipeline.

        Returns:
            MP3 bytes if ``config.return_bytes=True``, otherwise the output
            :class:`~pathlib.Path`.

        Raises:
            TTSError: On any TTS failure (source, engine, or pipeline level).
        """
        try:
            text = self.source.get_text()
            return self.engine.synthesize(text, self.config)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"TTS pipeline error: {exc}") from exc
