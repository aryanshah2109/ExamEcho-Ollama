"""
Application configuration loaded from environment variables / .env file.

All tunable parameters live here. Add new settings here rather than
hard-coding values inside service or AI modules.
"""

from __future__ import annotations

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq (cloud LLM)
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.0
    GROQ_MAX_TOKENS: int = 2048

    # Whisper
    WHISPER_MODEL_SIZE: str = "base"          # tiny | base | small | medium | large
    STT_DEFAULT_MODEL: str = "whisper"        # whisper | hf

    # MCQ Evaluation
    MCQ_EVAL_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    MCQ_SIMILARITY_THRESHOLD: float = 0.75

    # TTS
    TTS_AUDIO_DIR: str = "generated_audio"    # directory for temp TTS files

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # API
    API_V1_PREFIX: str = "/api/v1"
    APP_TITLE: str = "ExamEcho AI Service"
    APP_VERSION: str = "3.0.0"
    APP_DESCRIPTION: str = (
        "AI microservice powering ExamEcho: STT, TTS, "
        "question generation, rubric creation, and answer evaluation. "
        "Uses Groq (llama-3.3-70b-versatile) for all LLM tasks — fast cloud inference."
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
