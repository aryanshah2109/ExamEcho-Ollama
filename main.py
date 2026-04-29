"""
ExamEcho AI Service — FastAPI application entrypoint (Ollama edition).

Start the server:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Prerequisites:
    1. Ollama must be running:   ollama serve
    2. Model must be pulled:     ollama pull mistral:7b
    3. ffmpeg must be on PATH for audio conversion (STT)

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

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan: load all heavy models ONCE at startup ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Loads Whisper, Ollama (mistral:7b), and SentenceTransformer into
    ``app_state`` on startup so every request reuses already-loaded models.

    Startup failures are logged but do not abort the server — individual
    endpoints will return 503 if their required model is not ready.
    """
    logger.info("=" * 60)
    logger.info("Starting ExamEcho AI Service (Ollama edition) …")
    logger.info("  Ollama URL:   %s", settings.OLLAMA_BASE_URL)
    logger.info("  Ollama model: %s", settings.OLLAMA_MODEL_NAME)
    logger.info("  Whisper size: %s", settings.WHISPER_MODEL_SIZE)
    logger.info("=" * 60)

    # ── Whisper STT ──────────────────────────────────────────────────────────
    try:
        from ai_ml.model_creator import WhisperModelLoader
        app_state.whisper_model = WhisperModelLoader.get_model()
        logger.info("✓ Whisper (%s) model ready", settings.WHISPER_MODEL_SIZE)
    except Exception as exc:
        logger.error("✗ Whisper model failed to load: %s", exc)
        logger.warning("  STT endpoints will not be functional.")

    # ── Ollama LLM ───────────────────────────────────────────────────────────
    try:
        from ai_ml.model_creator import OllamaModelLoader
        app_state.ollama_model = OllamaModelLoader.get_model()
        logger.info(
            "✓ Ollama model '%s' ready at %s",
            settings.OLLAMA_MODEL_NAME,
            settings.OLLAMA_BASE_URL,
        )
    except Exception as exc:
        logger.error("✗ Ollama model failed to load: %s", exc)
        logger.warning(
            "  Question generation, rubric generation, and answer evaluation will not be functional.\n"
            "  Check that Ollama is running ('ollama serve') and the model is pulled "
            "('ollama pull %s').",
            settings.OLLAMA_MODEL_NAME,
        )

    # ── SentenceTransformer (MCQ evaluation) ─────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer '%s' …", settings.MCQ_EVAL_MODEL_NAME)
        app_state.st_model = SentenceTransformer(settings.MCQ_EVAL_MODEL_NAME)
        logger.info("✓ SentenceTransformer ready")
    except Exception as exc:
        logger.error("✗ SentenceTransformer failed to load: %s", exc)
        logger.warning("  MCQ evaluation endpoints will not be functional.")

    # ── Summary ──────────────────────────────────────────────────────────────
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


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

from app.routers import (  # noqa: E402
    evaluation,
    mcq_evaluation,
    question_generator,
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
]:
    app.include_router(router, prefix=settings.API_V1_PREFIX)


# ── Health check ──────────────────────────────────────────────────────────────

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
            "llm": "ollama",
            "model": settings.OLLAMA_MODEL_NAME,
            "ollama_url": settings.OLLAMA_BASE_URL,
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
    summary="Ollama server connectivity check",
    description=(
        "Probes the Ollama server directly and returns its status and "
        "whether the configured model is available locally."
    ),
)
def health_ollama() -> dict:
    """
    Live probe of the Ollama server.

    Unlike the main ``/health`` endpoint (which reflects startup state),
    this endpoint queries Ollama on every call.  Useful for debugging
    connectivity issues without restarting the service.
    """
    from ai_ml.model_creator import check_ollama_server, check_ollama_model
    from ai_ml.exceptions import OllamaConnectionError, ModelLoadError

    result: dict = {
        "ollama_url": settings.OLLAMA_BASE_URL,
        "model": settings.OLLAMA_MODEL_NAME,
        "server_reachable": False,
        "model_available": False,
        "error": None,
    }

    try:
        check_ollama_server(settings.OLLAMA_BASE_URL)
        result["server_reachable"] = True
    except OllamaConnectionError as exc:
        result["error"] = str(exc)
        return result

    try:
        check_ollama_model(settings.OLLAMA_MODEL_NAME, settings.OLLAMA_BASE_URL)
        result["model_available"] = True
    except ModelLoadError as exc:
        result["error"] = str(exc)

    return result
