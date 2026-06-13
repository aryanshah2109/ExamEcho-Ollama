"""
ExamEcho AI Service — FastAPI application entrypoint (Groq edition).

Start the server:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Prerequisites:
    1. GROQ_API_KEY must be set in .env
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

# Reconfigure stdout/stderr to UTF-8 encoding on Windows to prevent UnicodeEncodeError in logging
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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

    Loads Whisper, the Groq LLM wrapper, and SentenceTransformer into
    ``app_state`` on startup so every request reuses already-loaded models.

    Startup failures are logged but do not abort the server — individual
    endpoints will return 503 if their required model is not ready.
    """
    logger.info("=" * 60)
    logger.info("Starting ExamEcho AI Service (Groq edition) …")
    logger.info("  Groq model: %s", settings.GROQ_MODEL_NAME)
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

    # Groq LLM
    try:
        from ai_ml.model_creator import GroqModelLoader
        app_state.groq_model = GroqModelLoader.get_model()
        logger.info("✓ Groq model '%s' ready", settings.GROQ_MODEL_NAME)
    except Exception as exc:
        logger.error("✗ Groq model failed to load: %s", exc)
        logger.warning(
            "  Question generation, rubric generation, and answer evaluation will not be functional.\n"
            "  Check that GROQ_API_KEY is configured and the model name is valid."
        )

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
            ready.append("LLM (question gen / eval / rubrics)")
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
            "llm": "groq",
            "model": settings.GROQ_MODEL_NAME,
            "ollama_url": "https://api.groq.com/openai/v1",
        },
        "models": {
            "whisper": app_state.stt_ready,
            "ollama": app_state.llm_ready,
            "sentence_transformer": app_state.mcq_ready,
        },
    }


@app.get(
    "/health/ollama",
    tags=["Health"],
    summary="Groq connectivity check",
    description=(
        "Probes the configured Groq model wrapper and returns its readiness status."
    ),
)
def health_ollama() -> dict:
    """
    Live probe of the Groq configuration.

    Unlike the main ``/health`` endpoint (which reflects startup state),
    this endpoint validates the Groq configuration on every call.
    """
    from ai_ml.exceptions import GroqConnectionError, ModelLoadError
    from ai_ml.model_creator import GroqModelLoader

    result: dict = {
        "ollama_url": "https://api.groq.com/openai/v1",
        "model": settings.GROQ_MODEL_NAME,
        "server_reachable": False,
        "model_available": False,
        "error": None,
    }

    try:
        GroqModelLoader.get_model()
        result["server_reachable"] = True
        result["model_available"] = True
    except GroqConnectionError as exc:
        result["error"] = str(exc)
    except ModelLoadError as exc:
        result["error"] = str(exc)

    return result
