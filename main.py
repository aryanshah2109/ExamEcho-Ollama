"""
ExamEcho AI Service — FastAPI application entrypoint (OpenAI edition).

Start the server:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Prerequisites:
    1. OpenAI API key must be configured in the environment.
    2. ffmpeg must be on PATH for audio conversion (STT)

API docs available at:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.state import app_state

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# Lifespan: load all heavy models ONCE at startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Loads Whisper, OpenAI client, and SentenceTransformer into
    ``app_state`` on startup so every request reuses already-loaded clients/models.

    Startup failures are logged but do not abort the server — individual
    endpoints will return 503 if their required model is not ready.
    """
    logger.info("=" * 60)
    logger.info("Starting ExamEcho AI Service (OpenAI edition) …")
    logger.info("  OpenAI models: question=%s rubric=%s eval=%s mcq=%s",
                settings.OPENAI_MODEL_QUESTION,
                settings.OPENAI_MODEL_RUBRIC,
                settings.OPENAI_MODEL_EVAL,
                settings.OPENAI_MODEL_MCQ)
    logger.info("  Whisper size: %s", settings.WHISPER_MODEL_SIZE)
    logger.info("=" * 60)

    # Whisper STT
    try:
        from ai_ml.model_creator import WhisperModelLoader
        app_state.whisper_model = WhisperModelLoader.get_model()
        logger.info("✓ Whisper (%s) model ready", settings.WHISPER_MODEL_SIZE)
    except Exception as exc:
        logger.error("✗ Whisper model failed to load: %s", exc)
        logger.warning("  STT endpoints will not be functional.")

    # OpenAI LLM
    try:
        from ai_ml.model_creator import OpenAIClientLoader
        app_state.openai_client = OpenAIClientLoader.get_client()
        logger.info("✓ OpenAI client ready")
    except Exception as exc:
        logger.error("✗ OpenAI client failed to initialise: %s", exc)

    # LLM cache backend
    try:
        from app.utils.llm_cache import build_llm_cache_healthcheck
        cache_health = build_llm_cache_healthcheck(strict_redis=False)
        logger.info(
            "✓ LLM cache backend ready (%s, reachable=%s)",
            cache_health.get("backend"),
            cache_health.get("reachable"),
        )
        if cache_health.get("backend") == "sqlite":
            logger.info("  Using SQLite cache fallback at %s", cache_health.get("path"))
        elif not cache_health.get("reachable"):
            logger.warning("  Redis cache is not reachable; using fallback behavior where applicable.")
    except Exception as exc:
        logger.error("✗ LLM cache backend failed to initialise: %s", exc)

    # SentenceTransformer (MCQ evaluation)
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer '%s' …", settings.MCQ_EVAL_MODEL_NAME)
        app_state.st_model = SentenceTransformer(settings.MCQ_EVAL_MODEL_NAME)
        logger.info("✓ SentenceTransformer ready")
    except Exception as exc:
        logger.error("✗ SentenceTransformer failed to load: %s", exc)
        logger.warning("  MCQ evaluation endpoints will not be functional.")

    # Summary
    if app_state.is_ready:
        logger.info("✓ All models loaded — service is fully ready.")
    else:
        ready = []
        if app_state.stt_ready:
            ready.append("STT")
        if app_state.llm_ready:
            ready.append("LLM (question gen / eval / rubrics / MCQ)")
        if app_state.mcq_ready:
            ready.append("MCQ evaluation")
        logger.warning(
            "Service started in DEGRADED state. "
            "Functional: [%s]. Check logs above for errors.",
            ", ".join(ready) if ready else "none",
        )

    yield

    logger.info("Shutting down ExamEcho AI Service.")


# FastAPI app

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers

from app.routers import (  # noqa: E402
    evaluation,
    mcq_evaluation,
    question_generator,
    mcq_generator,
    rubrics,
    stt,
    tts,
)

for router in [
    stt.router,
    tts.router,
    evaluation.router,
    rubrics.router,
    mcq_evaluation.router,
    question_generator.router,
    mcq_generator.router,
]:
    app.include_router(router, prefix=settings.API_V1_PREFIX)


# Health check

@app.get(
    "/health",
    tags=["Health"],
    summary="Service health check",
    description=(
        "Returns the overall service status and per-model readiness. "
        "Use this for Kubernetes liveness / readiness probes."
    ),
)
def health_check() -> dict:
    """
    Returns HTTP 200 with model-readiness information.

    ``status`` is ``"ok"`` when all models are loaded, ``"degraded"``
    otherwise.  Individual model flags let the caller identify which
    capabilities are unavailable.
    """
    return {
        "status": "ok" if app_state.is_ready else "degraded",
        "version": settings.APP_VERSION,
        "backend": {
            "llm": "openai",
            "question_model": settings.OPENAI_MODEL_QUESTION,
            "rubric_model": settings.OPENAI_MODEL_RUBRIC,
            "evaluation_model": settings.OPENAI_MODEL_EVAL,
            "mcq_model": settings.OPENAI_MODEL_MCQ,
            "cache_enabled": settings.LLM_CACHE_ENABLED,
            "cache_backend": settings.LLM_CACHE_BACKEND,
            "cache_path": settings.LLM_CACHE_PATH,
        },
        "models": {
            "whisper": app_state.stt_ready,
            "openai": app_state.llm_ready,
            "sentence_transformer": app_state.mcq_ready,
        },
    }


@app.get(
    "/health/openai",
    tags=["Health"],
    summary="OpenAI backend configuration check",
    description=(
        "Returns the configured OpenAI backend model names and whether the "
        "client has been initialised."
    ),
)
def health_openai() -> dict:
    """
    Report the configured OpenAI backend without making a paid API call.
    """
    result: dict = {
        "client_initialized": app_state.llm_ready,
        "api_key_configured": bool(settings.OPENAI_API_KEY),
        "question_model": settings.OPENAI_MODEL_QUESTION,
        "rubric_model": settings.OPENAI_MODEL_RUBRIC,
        "evaluation_model": settings.OPENAI_MODEL_EVAL,
        "mcq_model": settings.OPENAI_MODEL_MCQ,
        "cache_enabled": settings.LLM_CACHE_ENABLED,
        "cache_backend": settings.LLM_CACHE_BACKEND,
        "cache_path": settings.LLM_CACHE_PATH,
        "error": None,
    }
    return result


@app.get(
    "/health/redis",
    tags=["Health"],
    summary="Redis cache health check",
    description=(
        "Returns Redis cache reachability and falls back details if Redis is "
        "not configured or not reachable."
    ),
)
def health_redis() -> dict:
    """
    Check whether the Redis cache backend is reachable.
    """
    from app.utils.llm_cache import build_llm_cache_healthcheck

    return build_llm_cache_healthcheck(strict_redis=True)
