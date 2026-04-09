# ---------------------------------------------------------------------------
# Build Stage
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

# Set working directory to the app root
WORKDIR /app

# 1. Copy only dependency files first to leverage Docker caching
COPY pyproject.toml uv.lock* ./

# 2. Install dependencies using uv
# This creates a virtual environment in /app/.venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

# 3. Copy the rest of the project files
# This includes models.py, openenv.yaml, and the server/ folder
COPY . .

# 4. Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# ---------------------------------------------------------------------------
# Runtime Stage
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment and all project files from the builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Set environment paths to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Use port 7860 for Hugging Face Spaces compatibility
EXPOSE 7860

ENV PORT=8000

# Metadata validation: ensures openenv validate can find everything
LABEL org.openenv.version="1.0.0"

# Healthcheck to ensure the FastAPI server is responsive
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

# Start the server using the entry point defined in pyproject.toml
# We override the port to 7860 for the HF Space environment
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
