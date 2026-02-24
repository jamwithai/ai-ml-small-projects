# ---- Base image ----
ARG BASE_IMAGE=vllm/vllm-openai:v0.11.0
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install the FastAPI wrapper deps (vllm/torch come from the base image)
RUN uv pip install --system --no-cache --break-system-packages \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.34" \
    "pydantic-settings>=2.7"

# Install the project (no-deps because vllm/torch are already satisfied)
COPY pyproject.toml .python-version ./
COPY src/ src/
RUN uv pip install --system --no-cache --break-system-packages --no-deps .

EXPOSE 8000
CMD ["python", "-m", "vllm_serve.app"]
