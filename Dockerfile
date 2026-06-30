# =============================================================================
# Unified Dockerfile — Agent, UI, and MCP
# =============================================================================
# Usage: Docker-compose passes the TARGET arg (agent, ui, or mcp)
# =============================================================================

# Declare the target service (defaults to agent)
ARG TARGET=agent

# -----------------------------------------------------------------------------
# Stage 1: Base Builder (Installs Python packages for ALL services)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Model Downloader (ONLY runs if TARGET=agent)
# This prevents the UI and MCP from downloading 2.2GB of ML models!
# -----------------------------------------------------------------------------
FROM builder AS model-downloader
ARG TARGET

RUN mkdir -p /app/models

RUN if [ "$TARGET" = "agent" ]; then \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5'); print('Embedder downloaded')" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512); print('Reranker downloaded')"; \
    fi

# -----------------------------------------------------------------------------
# Stage 3: Final Production Image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS production

WORKDIR /app
ARG TARGET

# CRITICAL FIX: ARGs disappear after build. ENV keeps them alive at runtime!
ENV TARGET=${TARGET}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy Python packages from Stage 1
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy models ONLY if we built the agent (Stage 2)
COPY --from=model-downloader /app/models /app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Copy ALL application code (docker-compose.yml controls what gets mounted over this)
COPY . /app/

RUN mkdir -p /app/logs /app/data /app/artifacts

# Start the correct service based on the TARGET argument
# NOTE: If your UI main file is named main.py instead of app.py, change it below!
CMD if [ "$TARGET" = "agent" ]; then \
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1; \
    elif [ "$TARGET" = "ui" ]; then \
        streamlit run /app/ui/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true; \
    elif [ "$TARGET" = "mcp" ]; then \
        python -m mcp_server.main; \
    fi