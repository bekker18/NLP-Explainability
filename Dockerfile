# syntax=docker/dockerfile:1

# ------------------------------------------------------------------
# Fake News Detector + SHAP — Gradio demo image
# Python 3.12 to match the project (cpython-312 bytecode in the repo).
# ------------------------------------------------------------------
FROM python:3.12-slim

# Sensible Python defaults inside containers
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  # Gradio must bind to all interfaces to be reachable from the host,
  # and we pin the port the rest of this file exposes.
  GRADIO_SERVER_NAME=0.0.0.0 \
  GRADIO_SERVER_PORT=7860 \
  # Writable cache locations for the non-root user
  MPLCONFIGDIR=/tmp/matplotlib \
  HF_HOME=/tmp/huggingface

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
# build-essential is installed only to compile any sdist that lacks a wheel,
# then purged to keep the final image small.
COPY requirements.txt ./
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential \
  && pip install --no-cache-dir -r requirements.txt \
  && apt-get purge -y --auto-remove build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy the application code and the artefacts the demo needs at startup
# (models/pipeline.joblib and data/X_test.csv). The large raw training
# CSVs are excluded via .dockerignore.
COPY . .

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser \
  && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Basic liveness check against the Gradio server
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/').status==200 else 1)" || exit 1

CMD ["python", "app/demo.py"]