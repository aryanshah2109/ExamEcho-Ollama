"""
Application configuration loaded from environment variables / .env file.

All tunable parameters live here. Add new settings here rather than
hard-coding values inside service or AI modules.
"""

from __future__ import annotations

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI (hosted LLM)
    OPENAI_API_KEY: str = ""
    OPENAI_ORGANIZATION_ID: str | None = None
    OPENAI_PROJECT_ID: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_MODEL_QUESTION: str = "gpt-5.4-nano"
    OPENAI_MODEL_RUBRIC: str = "gpt-5.4-nano"
    OPENAI_MODEL_EVAL: str = "gpt-5.4-mini"
    OPENAI_MODEL_MCQ: str = "gpt-5.4-nano"
    OPENAI_TEMPERATURE: float = 0.0
    OPENAI_MAX_OUTPUT_TOKENS_QUESTION: int = 1200
    OPENAI_MAX_OUTPUT_TOKENS_RUBRIC: int = 300
    OPENAI_MAX_OUTPUT_TOKENS_EVAL: int = 700
    OPENAI_MAX_OUTPUT_TOKENS_MCQ: int = 1400
    OPENAI_TIMEOUT_SECONDS: float = 30.0
    OPENAI_MAX_RETRIES: int = 1
    OPENAI_GENERATION_CHUNK_SIZE: int = 10
    OPENAI_TOPIC_BATCH_SIZE: int = 4

    # LLM cache
    LLM_CACHE_ENABLED: bool = True
    LLM_CACHE_BACKEND: str = "redis"   # redis | sqlite
    REDIS_URL: str = ""
    REDIS_CACHE_PREFIX: str = "examecho:llm:"
    REDIS_CONNECT_RETRIES: int = 3
    REDIS_CONNECT_BACKOFF_SECONDS: float = 0.5
    REDIS_CONNECT_TIMEOUT_SECONDS: float = 2.0
    LLM_CACHE_PATH: str = ".cache/llm_cache.sqlite3"
    LLM_CACHE_TTL_SECONDS: int = 7 * 24 * 60 * 60

    # Whisper
    WHISPER_MODEL_SIZE: str = "base"          # tiny | base | small | medium | large
    STT_DEFAULT_MODEL: str = "whisper"        # whisper | hf
    STT_VAD_ENABLED: bool = False

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
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = (
        "AI microservice powering ExamEcho: STT, TTS, "
        "question generation, rubric creation, and answer evaluation. "
        "Uses OpenAI for all LLM tasks."
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
