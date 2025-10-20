# =============================================================================
# UNDERDOG Trading System - Production Dockerfile
# Python 3.13 + MetaTrader 5 + ML/Monitoring Stack
# =============================================================================

FROM python:3.13-slim

# Metadata
LABEL maintainer="whateve-r"
LABEL description="UNDERDOG Algorithmic Trading System"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry --version

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry install --no-root --no-dev \
    && pip install prometheus-client fastapi uvicorn requests \
    && rm -rf ~/.cache/pip

# Copy application code
COPY underdog/ ./underdog/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY README.md ./

# Create necessary directories
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/historical \
    /app/logs \
    /app/mlruns \
    /app/models

# Set permissions
RUN chmod +x scripts/*.py

# Expose ports
# 8000 - FastAPI monitoring dashboard
# 8080 - MLflow UI (if running)
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Default command (can be overridden in docker-compose.yml)
CMD ["python", "scripts/start_live.py"]
