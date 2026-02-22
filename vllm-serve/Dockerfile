FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cache-friendly)
COPY pyproject.toml .python-version ./
RUN uv sync --no-dev --no-install-project

# Copy source and install project
COPY src/ src/
RUN uv sync --no-dev

EXPOSE 8000

# Production: use multiple workers behind the vLLM engine
CMD ["uv", "run", "vllm-serve"]
