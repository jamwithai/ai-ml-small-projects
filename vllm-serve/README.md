# vllm-serve

A minimal, production-ready LLM serving stack using **vLLM** and **FastAPI**.

## Why vLLM?

Serving LLMs naively (one request at a time, basic sampling) wastes GPU memory and is painfully slow. vLLM solves this with two key ideas:

- **PagedAttention** — manages GPU memory like an OS manages RAM. Instead of pre-allocating massive contiguous blocks for each request's KV cache, it pages them dynamically. This means you can serve **2-4x more concurrent requests** on the same GPU.
- **Continuous batching** — instead of waiting for an entire batch to finish before starting the next, vLLM slots new requests into the batch as soon as a slot opens. This keeps the GPU saturated and slashes time-to-first-token for queued requests.

vLLM also handles tensor parallelism (split one model across multiple GPUs), quantization (AWQ, GPTQ, etc.), and speculative decoding — all behind a single `AsyncLLMEngine` interface.

## Why FastAPI on top?

vLLM has a built-in server, but wrapping it in FastAPI gives you full control over the API layer:

- **Custom endpoints** — add auth, rate limiting, request validation, logging, or any middleware you need
- **OpenAI-compatible API** — `/v1/chat/completions` with streaming, so any OpenAI SDK client works out of the box
- **Health checks** — `/health` endpoint for Kubernetes liveness/readiness probes
- **Lifespan management** — the engine loads once at startup and shuts down cleanly, not per-request

This is the standard pattern in production: vLLM handles the GPU and inference, FastAPI handles everything between the client and the engine.

## Project structure

```
src/vllm_serve/
├── config.py    # All config via VLLM_* environment variables
├── models.py    # OpenAI-compatible request/response schemas
├── engine.py    # vLLM engine startup + sampling params
└── app.py       # FastAPI app, routes, streaming
```

## Model download

Models are downloaded automatically from HuggingFace Hub the first time vLLM loads them. Weights and tokenizer files are cached in `~/.cache/huggingface/hub/`, so subsequent starts are fast.

vLLM supports most popular HuggingFace architectures — Llama, Mistral, Mixtral, Qwen, Gemma, Phi, DeepSeek, Command R, and many more including vision-language models. See the full list at [vLLM Supported Models](https://docs.vllm.ai/en/stable/models/supported_models/).

For gated models (like Llama), you need to accept the license on HuggingFace and authenticate:

```bash
# Option 1: login interactively (saves token to ~/.cache/huggingface/token)
huggingface-cli login

# Option 2: set the token directly
export HF_TOKEN=hf_your_token_here
```

## Quickstart

```bash
# Install
uv sync

# Configure
cp .env.example .env
# Edit .env with your model, HF_TOKEN, etc.

# Run (requires a GPU)
uv run vllm-serve

# Chat (streaming)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

## Interactive docs

FastAPI auto-generates interactive API docs. Once the server is running:

- **Swagger UI** → [http://localhost:8000/docs](http://localhost:8000/docs) — try out requests directly from the browser
- **ReDoc** → [http://localhost:8000/redoc](http://localhost:8000/redoc) — cleaner read-only view

You can use Swagger UI to send requests to `/v1/chat/completions` without needing `curl` or any client code. Set `"stream": false` in the request body to see the full response inline.

## Configuration

All settings are environment variables with the `VLLM_` prefix:

| Variable | Default | Description |
|---|---|---|
| `VLLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of GPU memory to use |
| `VLLM_QUANTIZATION` | `None` | Quantization method (awq, gptq, etc.) |
| `VLLM_MAX_MODEL_LEN` | `None` | Override max sequence length |
| `VLLM_DTYPE` | `auto` | Model dtype (auto, float16, bfloat16) |
| `VLLM_HOST` | `0.0.0.0` | Server bind address |
| `VLLM_PORT` | `8000` | Server port |

## Docker

```bash
docker build -t vllm-serve .
docker run --gpus all -p 8000:8000 \
  -e VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  vllm-serve
```
