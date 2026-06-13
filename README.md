# ExamEcho AI Service

AI microservice powering the ExamEcho examination platform.

LLM inference now runs through the OpenAI API with structured outputs,
persistent caching, and smaller model defaults to keep costs down.

Provides REST endpoints for:
- STT - Speech-to-Text (OpenAI Whisper, local)
- TTS - Text-to-Speech (gTTS)
- Question Generation - topic/difficulty-based exam questions
- Rubric Generation - marking criteria from question + marks
- Answer Evaluation - viva/long-answer scoring with feedback
- MCQ Evaluation - option matching via cosine similarity

## Prerequisites

| Dependency | Notes |
|---|---|
| Python | 3.10+ recommended |
| OpenAI API key | Required for all LLM endpoints |
| ffmpeg | Required for audio conversion (STT) |

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Copy your environment file and set the OpenAI values:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | empty | OpenAI API key |
| `OPENAI_MODEL_QUESTION` | `gpt-5.4-nano` | Model for question generation |
| `OPENAI_MODEL_RUBRIC` | `gpt-5.4-nano` | Model for rubric generation |
| `OPENAI_MODEL_EVAL` | `gpt-5.4-mini` | Model for answer evaluation |
| `OPENAI_MODEL_MCQ` | `gpt-5.4-nano` | Model for MCQ generation |
| `OPENAI_TEMPERATURE` | `0.0` | Deterministic output |
| `OPENAI_MAX_OUTPUT_TOKENS_*` | task-specific | Per-task output limits |
| `OPENAI_MAX_RETRIES` | `1` | Keep retries low to control spend |
| `OPENAI_GENERATION_CHUNK_SIZE` | `10` | Splits large generation requests into smaller calls |
| `OPENAI_TOPIC_BATCH_SIZE` | `4` | Batches multiple topics into one paid request when possible |
| `LLM_CACHE_ENABLED` | `true` | Enables persistent structured-response caching |
| `LLM_CACHE_BACKEND` | `redis` | `redis` or `sqlite` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis cache endpoint |
| `REDIS_CACHE_PREFIX` | `examecho:llm:` | Key prefix for Redis cache entries |
| `REDIS_CONNECT_RETRIES` | `3` | Redis connection retries at startup |
| `REDIS_CONNECT_BACKOFF_SECONDS` | `0.5` | Backoff between Redis connection retries |
| `REDIS_CONNECT_TIMEOUT_SECONDS` | `2.0` | Redis socket timeout |
| `LLM_CACHE_PATH` | `.cache/llm_cache.sqlite3` | SQLite cache location |
| `LLM_CACHE_TTL_SECONDS` | `604800` | Cache retention window |
| `WHISPER_MODEL_SIZE` | `base` | `tiny` \| `base` \| `small` \| `medium` \| `large` |
| `STT_DEFAULT_MODEL` | `whisper` | `whisper` \| `hf` |
| `MCQ_EVAL_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | SentenceTransformer model |
| `MCQ_SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity threshold |
| `TTS_AUDIO_DIR` | `generated_audio` | Directory for temp MP3 files |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `API_V1_PREFIX` | `/api/v1` | API route prefix |

## Run Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The service will:
1. Load Whisper and SentenceTransformer locally.
2. Initialize the OpenAI client from environment variables.
3. Create or reuse the local SQLite cache for structured LLM responses.
4. Serve API docs at `http://localhost:8000/docs`.

## Run with Docker

```bash
docker compose up -d
```

The compose stack starts both the app and a Redis cache. Make sure
`OPENAI_API_KEY` is set in your shell or `.env` before starting the stack.

## Health

### General health

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "version": "2.0.0",
  "backend": {
    "llm": "openai",
    "question_model": "gpt-5.4-nano",
    "rubric_model": "gpt-5.4-nano",
    "evaluation_model": "gpt-5.4-mini",
    "mcq_model": "gpt-5.4-nano",
    "cache_enabled": true,
    "cache_backend": "redis",
    "cache_path": ".cache/llm_cache.sqlite3"
  },
  "models": {
    "whisper": true,
    "openai": true,
    "sentence_transformer": true
  }
}
```

### OpenAI backend configuration

```bash
curl http://localhost:8000/health/openai
```

This endpoint reports whether the OpenAI client was initialized and whether
the API key is configured, without making a paid network call.

### Redis cache health

```bash
curl http://localhost:8000/health/redis
```

This endpoint reports whether Redis is reachable and whether the service
is using Redis or the SQLite fallback.

## Cost Control

The implementation uses several cost-saving measures:
- Structured outputs to reduce malformed JSON retries
- Small default models for generation tasks
- Slightly stronger model only for answer evaluation
- Redis caching for cross-worker identical-request deduplication
- Chunked generation for large question/MCQ requests
- Low retry count to avoid duplicate paid requests

For offline or back-office bulk generation, prefer the OpenAI Batch API.
It is a better fit when you do not need immediate responses.

## Project Layout

```text
.
├── ai_ml/
│   ├── evaluation.py
│   ├── mcq_generator.py
│   ├── model_creator.py
│   ├── openai_llm.py
│   ├── question_generator.py
│   └── rubrics.py
├── app/
│   ├── config.py
│   ├── core/state.py
│   ├── services/
│   └── utils/llm_cache.py
├── docker-compose.yml
├── Dockerfile
└── main.py
```
