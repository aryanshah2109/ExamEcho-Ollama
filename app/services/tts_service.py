"""
TTS service: bridges FastAPI route → TTS AI module.

Generates an MP3 audio file for a question string and returns its path.
The caller (router) is responsible for serving the file; after the
response is sent, the file should be deleted via a BackgroundTask.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ai_ml.tts import GTTSEngine, DirectTextSource, TTSConfig, TTSPipeline
from app.config import settings

logger = logging.getLogger(__name__)


def generate_speech(*, text: str, question_id: str, language: str = "en", slow: bool = False) -> Path:
    """
    Synthesise speech from text and write to a named MP3 file.

    Args:
        text:        Text to convert to speech.
        question_id: Used to name the output file (``<question_id>.mp3``).
        language:    BCP-47 language code (default ``"en"``).
        slow:        If ``True``, use slower speech rate.

    Returns:
        :class:`~pathlib.Path` of the generated MP3 file.

    Raises:
        TTSError: If synthesis fails.
    """
    audio_dir = Path(settings.TTS_AUDIO_DIR)
    audio_dir.mkdir(parents=True, exist_ok=True)

    output_path = audio_dir / f"{question_id}.mp3"

    config = TTSConfig(
        language=language,
        slow=slow,
        output_file=output_path,
        return_bytes=False,
    )

    pipeline = TTSPipeline(
        source=DirectTextSource(text),
        engine=GTTSEngine(),
        config=config,
    )

    result_path = pipeline.run()
    logger.info("TTS audio written to %s", result_path)
    return Path(result_path)


def delete_audio_file(path: str | Path) -> None:
    """
    Remove a generated audio file from disk.

    Intended for use as a FastAPI ``BackgroundTask`` so temp files are
    cleaned up after the response has been streamed to the client.

    Args:
        path: Path to the file to delete.
    """
    try:
        os.unlink(path)
        logger.debug("Deleted TTS audio file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete TTS audio file %s: %s", path, exc)
