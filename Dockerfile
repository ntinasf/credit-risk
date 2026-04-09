# Dockerfile

# ── Stage 1: Build environment ────────────────────────────────────────────────
FROM python:3.13-slim AS builder
WORKDIR /app

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (Docker layer caching)
# These change far less often than source code.
# If only source changes, Docker skips this expensive layer.
COPY pyproject.toml uv.lock README.md ./

# Install dependencies to a virtual environment (include the 'app' extra for streamlit)
RUN uv sync --no-dev --extra app --frozen

# Copy the package source and install it
COPY src/ src/
RUN uv pip install --no-cache -e .

# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime
WORKDIR /app

# Copy only the virtual environment from builder (not build tools)
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Copy application code
COPY app/ app/

# Make the venv's executables available
ENV PATH="/app/.venv/bin:$PATH"

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]