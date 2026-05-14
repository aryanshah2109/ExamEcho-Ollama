"""
STT service: bridges FastAPI route → STT AI module.

Handles temp file I/O, delegates transcription to :mod:`ai_ml.stt`,
and ensures cleanup of uploaded audio files after processing.
"""

from __future__ import annotations

import logging
import os
import tempfile

from fastapi import UploadFile

from ai_ml.stt import STT
from ai_ml.audio_preprocessor import AudioPreprocessor, AudioPreprocessorConfig
from app.config import settings
from app.core.state import app_state

logger = logging.getLogger(__name__)

# Audio MIME types accepted for transcription
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/webm",
    "audio/ogg",
}

config = AudioPreprocessorConfig(vad_enabled=settings.STT_VAD_ENABLED)
preprocessor = AudioPreprocessor(config=config)

async def transcribe_audio(audio: UploadFile, lang: str = "en", model: str | None = None) -> str:
    chosen_model = model or settings.STT_DEFAULT_MODEL
    suffix = _extension_from_content_type(audio.content_type)
    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await audio.read()
            tmp.write(content)

        # Use module-level preprocessor (VAD disabled per config)
        result = preprocessor.preprocess_file(tmp_path)
        processed_path = result.metadata.processed_path

        if chosen_model == "whisper" and app_state.whisper_model is not None:
            output = app_state.whisper_model.transcribe(processed_path, language=lang)
            text = output["text"].strip() if isinstance(output, dict) else ""
        else:
            stt = STT(lang=lang, model=chosen_model, audio_file_path=processed_path)
            text = stt.transcribe()

        return text or ""

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def _extension_from_content_type(content_type: str | None) -> str:
    """Map MIME type to a file extension for temp file naming."""
    mapping = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".mp4",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
    }
    return mapping.get(content_type or "", ".wav")
