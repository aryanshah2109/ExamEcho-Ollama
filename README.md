# ExamEcho AI Service — Ollama Edition

AI microservice powering the ExamEcho examination platform.  
All LLM inference runs **locally** via [Ollama](https://ollama.ai) using **mistral:7b** — no external API key required.

Provides REST endpoints for:
- **STT** — Speech-to-Text (OpenAI Whisper, local)
- **TTS** — Text-to-Speech (gTTS)
- **Question Generation** — Topic/difficulty-based question generation (mistral:7b via Ollama)
- **Rubric Generation** — Automatic marking criteria from question + marks (mistral:7b)
- **Answer Evaluation** — Viva/long-answer scoring with feedback (mistral:7b)
- **MCQ Evaluation** — Option matching via cosine similarity (SentenceTransformer, local)


---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| Ollama | Latest | [Install from ollama.ai](https://ollama.ai) |
| mistral:7b | — | Pull with `ollama pull mistral:7b` (~4 GB) |
| ffmpeg | Any recent | Required for audio conversion (STT) |

### Install Ollama

```bash
# Linux / WSL
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Windows
# Download installer from https://ollama.ai/download
```

### Pull the model

```bash
ollama pull mistral:7b
```

> The first pull downloads ~4 GB. Subsequent starts are instant.

### Start Ollama

```bash
ollama serve
# Runs on http://localhost:11434 by default
```

### Install ffmpeg

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

---

## Quick Start (Local)

```bash
# 1. Clone / extract the project
cd examecho_ai

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and review environment config
cp .env.example .env
# Default values work if Ollama is running on localhost:11434

# 5. Ensure Ollama is running and model is pulled
ollama serve &          # (skip if already running as a service)
ollama pull mistral:7b  # (skip if already pulled)

# 6. Start the service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The service will:
1. Probe Ollama at startup and verify `mistral:7b` is available.
2. Load Whisper and SentenceTransformer into memory.
3. Serve API docs at **http://localhost:8000/docs** (Swagger UI).
4. Serve ReDoc at **http://localhost:8000/redoc**.

### HuggingFace STT backend (optional)

If you want to use the `hf` STT backend:

```bash
pip install transformers torch
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL_NAME` | `mistral:7b` | Model to use for all LLM tasks |
| `OLLAMA_TEMPERATURE` | `0.0` | Output randomness (0 = deterministic) |
| `OLLAMA_MAX_TOKENS` | `2048` | Max tokens per LLM response |
| `WHISPER_MODEL_SIZE` | `base` | `tiny` \| `base` \| `small` \| `medium` \| `large` |
| `STT_DEFAULT_MODEL` | `whisper` | `whisper` \| `hf` |
| `MCQ_EVAL_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | SentenceTransformer model |
| `MCQ_SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity threshold for correct MCQ |
| `TTS_AUDIO_DIR` | `generated_audio` | Directory for temp MP3 files |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `API_V1_PREFIX` | `/api/v1` | API route prefix |

---

## Running with Docker Compose (Recommended)

The included `docker-compose.yml` starts **both** the Ollama server and the ExamEcho AI service:

```bash
# 1. Start all services
docker compose up -d

# 2. Pull the model inside the Ollama container (first time only, ~4 GB)
docker compose exec ollama ollama pull mistral:7b

# 3. Check readiness
curl http://localhost:8000/health

# 4. Open API docs
open http://localhost:8000/docs
```

### GPU support (Nvidia)

Uncomment the `deploy.resources` block in `docker-compose.yml` to enable GPU
acceleration (requires `nvidia-container-toolkit`).

---

## Running with Docker (manual)

```bash
# 1. Start Ollama container
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama

# 2. Pull the model
docker exec ollama ollama pull mistral:7b

# 3. Build the ExamEcho image
docker build -t examecho-ai .

# 4. Run the service (connected to the Ollama container)
docker run -d \
  --name examecho-ai \
  -p 8000:8000 \
  --link ollama:ollama \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  examecho-ai
```

---

## Health Checks

### General health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "2.0.0",
  "backend": {
    "llm": "ollama",
    "model": "mistral:7b",
    "ollama_url": "http://localhost:11434"
  },
  "models": {
    "whisper": true,
    "ollama": true,
    "sentence_transformer": true
  }
}
```

### Ollama connectivity check

```bash
curl http://localhost:8000/health/ollama
```

```json
{
  "ollama_url": "http://localhost:11434",
  "model": "mistral:7b",
  "server_reachable": true,
  "model_available": true,
  "error": null
}
```

---

## API Reference

All endpoints are prefixed with `/api/v1`.  
Full interactive documentation is available at **http://localhost:8000/docs**.

---

### Speech-to-Text

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/stt/transcribe` | Transcribe uploaded audio to text |

**Form params:** `audio` (file), `lang` (default `"en"`), `model` (default `"whisper"`)

**Accepted audio types:** `audio/wav`, `audio/x-wav`, `audio/mpeg`, `audio/mp4`, `audio/webm`, `audio/ogg`

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/stt/transcribe \
  -F "audio=@answer.webm" \
  -F "lang=en"
```

**Response:**
```json
{ "text": "A binary search tree is a tree where...", "language": "en", "model": "whisper" }
```

---

### Text-to-Speech

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/tts/synthesize` | Convert text to MP3 audio |

**Request body:**
```json
{
  "question_id": "q_001",
  "text": "What is polymorphism?",
  "language": "en",
  "slow": false
}
```

**Response:** MP3 audio file (binary stream).

---

### Question Generation

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/questions/generate` | Generate questions for one or more topics |

**Request body:**
```json
{
  "topics": ["Binary Trees", "Sorting Algorithms"],
  "num_questions": 5,
  "difficulty": "medium"
}
```

**Response:**
```json
{
  "topics": {
    "Binary Trees": {
      "1": "What is a binary search tree?",
      "2": "Explain the difference between pre-order and in-order traversal."
    }
  }
}
```

> **Note:** `mistral:7b` is slower than a cloud API. Expect ~5–15 s per topic on CPU,
> ~2–5 s with a GPU.

---

### Rubric Generation

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/rubrics/create` | Generate marking criteria for a question |

**Request body:**
```json
{
  "question_id": "q_001",
  "question_text": "Explain the difference between stack and queue.",
  "max_marks": 10
}
```

**Response:**
```json
{
  "question_id": "q_001",
  "question_text": "Explain the difference between stack and queue.",
  "rubrics": [
    "Correctly defines stack as LIFO (Last In, First Out).",
    "Correctly defines queue as FIFO (First In, First Out).",
    "Identifies at least one real-world use case for each structure.",
    "Describes the key operations (push/pop for stack, enqueue/dequeue for queue)."
  ]
}
```

---

### Answer Evaluation

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/evaluate/answer` | Evaluate a viva / long-form answer |

**Request body:**
```json
{
  "question_id": "q_001",
  "question_text": "What is a binary search tree?",
  "student_answer": "A BST is a tree where each node has at most two children and left child is smaller...",
  "rubric": ["Correct definition", "Mention of ordering property", "Example given"],
  "max_marks": 10
}
```

**Response:**
```json
{
  "question_id": "q_001",
  "score": 7,
  "strengths": ["Correct definition of BST", "Mentioned ordering property"],
  "weakness": ["No concrete example provided"],
  "justification": "The student demonstrated understanding of BST structure but lacked an example.",
  "suggested_improvement": "Include a concrete example diagram or walkthrough of BST insertion."
}
```

---

### MCQ Evaluation

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/mcq/evaluate` | Evaluate a multiple-choice answer |

**Request body:**
```json
{
  "question_id": "q_002",
  "selected_option": "option b",
  "correct_option": "option b"
}
```

**Response:**
```json
{
  "question_id": "q_002",
  "similarity_score": 1.0,
  "inference": "Correct Answer"
}
```

---

## Troubleshooting

### Ollama server not reachable

```
OllamaConnectionError: Cannot connect to Ollama at 'http://localhost:11434'.
Make sure Ollama is running:  ollama serve
```

**Fix:** Start Ollama with `ollama serve` or check it is installed.

---

### Model not found

```
ModelLoadError: Ollama model 'mistral:7b' is not available locally.
Pull the model first:  ollama pull mistral:7b
```

**Fix:** Run `ollama pull mistral:7b`. The download is ~4 GB.

---

### Slow responses

mistral:7b on CPU generates ~15–30 tokens/second, which means a 5-question
generation request takes ~30–60 seconds.  To speed this up:

- Use a GPU machine and set `OLLAMA_NUM_GPU=1` in your environment.
- Use a smaller model: change `OLLAMA_MODEL_NAME=mistral:7b-instruct-q4_K_M`.
- Run Ollama on a machine with ≥ 8 GB RAM for CPU inference.

---

### JSON parse errors from the LLM

mistral:7b occasionally adds prose preambles before the JSON.  The service
includes a robust JSON extractor (`app/utils/json_utils.py`) that strips
markdown fences and repairs common issues.  If errors persist:

- Lower `OLLAMA_TEMPERATURE` to `0.0` (already the default).
- Check the server logs for the raw model output.

---

## Project Structure

```
examecho_ai/
├── ai_ml/                         # Pure AI/ML logic (no FastAPI dependencies)
│   ├── exceptions.py              # All custom exceptions (incl. OllamaConnectionError)
│   ├── model_creator.py           # Singleton loaders: WhisperModelLoader, OllamaModelLoader
│   ├── audio_preprocessor.py      # ffmpeg conversion + VAD silence trimming
│   ├── stt.py                     # Speech-to-Text (Whisper / HF)
│   ├── tts.py                     # Text-to-Speech (gTTS pipeline)
│   ├── evaluation.py              # Viva answer evaluation (mistral:7b)
│   ├── rubrics.py                 # Rubric generation (mistral:7b)
│   ├── question_generator.py      # Exam question generation (mistral:7b)
│   └── mcq_evaluation.py          # MCQ semantic similarity evaluation
│
├── app/
│   ├── config.py                  # All settings (env-driven via pydantic-settings)
│   ├── core/
│   │   └── state.py               # Global app state (ollama_model, whisper_model, st_model)
│   ├── routers/                   # FastAPI route handlers
│   ├── schemas/                   # Pydantic request/response models
│   ├── services/                  # Business logic bridging routers ↔ ai_ml
│   └── utils/
│       └── json_utils.py          # Shared LLM JSON extraction/repair
│
├── main.py                        # FastAPI app, lifespan, router registration
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml             # Runs Ollama + ExamEcho AI together
└── README.md
```

---

## Performance Notes

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8+ cores |
| GPU | None (CPU inference) | Nvidia GPU with 6 GB+ VRAM |
| Disk | 10 GB free | 20 GB free |

CPU inference is usable but slow (~30–60 s per LLM request).  
GPU inference brings this down to ~2–5 s per request.
