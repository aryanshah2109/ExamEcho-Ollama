# ExamEcho AI Service - Dockerfile (OpenAI edition)
#
# This container runs only the FastAPI service.
# OpenAI credentials are supplied via environment variables or secrets.

FROM python:3.11-slim

# System deps: ffmpeg (audio conversion) + build tools for webrtcvad
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create runtime directories
RUN mkdir -p generated_audio
RUN mkdir -p .cache

# Expose service port
EXPOSE 8000

# Healthcheck - waits for the service to become healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

