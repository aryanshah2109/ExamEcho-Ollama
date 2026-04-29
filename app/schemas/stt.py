"""
Request / response schemas for the STT endpoint.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class STTResponse(BaseModel):
    """
    Response body for ``POST /api/v1/stt/transcribe``.

    Attributes:
        text:     Transcribed text (empty string if no speech detected).
        language: BCP-47 language code used for transcription.
        model:    STT backend used (``"whisper"`` or ``"hf"``).
    """

    text: str = ""
    language: Optional[str] = "en"
    model: Optional[str] = "whisper"
