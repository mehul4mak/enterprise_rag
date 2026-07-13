# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# System deps: PyMuPDF wheels are self-contained; we just need build-less runtime.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models

WORKDIR /app

# 1) Install Python deps first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Pre-download the embedding + reranker models so container startup is fast and offline.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("models cached at /models")
PY

# 3) App code.
COPY src ./src
COPY main.py ./

# Non-root user for safety. Pre-create the cache dir so the named volume inherits
# appuser ownership on first mount (otherwise Docker creates it root-owned = unwritable).
RUN useradd -m appuser && mkdir -p /app/.cache && chown -R appuser /app /models
USER appuser

EXPOSE 8000

# Healthcheck hits the API liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

# Default: serve the API. Override CMD for the CLI (python main.py --pdf ...).
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
