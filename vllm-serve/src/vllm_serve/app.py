from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from vllm_serve.config import settings
from vllm_serve.engine import build_sampling_params, get_engine, get_tokenizer, init_engine, shutdown_engine
from vllm_serve.models import (
    ChatRequest,
    ChatResponse,
    Choice,
    DeltaMessage,
    Message,
    StreamChoice,
    StreamChunk,
    Usage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vllm_serve")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Loading model: %s", settings.model)
    await init_engine()
    logger.info("Engine ready")
    yield
    logger.info("Shutting down")
    shutdown_engine()


app = FastAPI(title="vllm-serve", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _apply_chat_template(messages: list[Message]) -> str:
    tokenizer = get_tokenizer()
    return tokenizer.apply_chat_template(
        [m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True,
    )


def _request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": settings.model, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    engine = get_engine()
    prompt = _apply_chat_template(req.messages)
    params = build_sampling_params(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        stream=req.stream,
    )
    rid = _request_id()

    if req.stream:
        return StreamingResponse(
            _stream(engine, prompt, params, rid), media_type="text/event-stream"
        )

    final = None
    async for output in engine.generate(prompt=prompt, sampling_params=params, request_id=rid):
        final = output

    if final is None:
        raise HTTPException(status_code=500, detail="No output from engine")

    text = final.outputs[0].text
    return ChatResponse(
        id=rid,
        created=int(time.time()),
        model=req.model,
        choices=[Choice(message=Message(role="assistant", content=text), finish_reason="stop")],
        usage=Usage(
            prompt_tokens=len(final.prompt_token_ids),
            completion_tokens=len(final.outputs[0].token_ids),
            total_tokens=len(final.prompt_token_ids) + len(final.outputs[0].token_ids),
        ),
    )


async def _stream(engine, prompt: str, params, rid: str) -> AsyncIterator[str]:
    created = int(time.time())

    async for output in engine.generate(prompt=prompt, sampling_params=params, request_id=rid):
        text = output.outputs[0].text
        if text:
            chunk = StreamChunk(
                id=rid,
                created=created,
                model=settings.model,
                choices=[StreamChoice(delta=DeltaMessage(content=text))],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        if output.finished:
            break

    chunk = StreamChunk(
        id=rid,
        created=created,
        model=settings.model,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def cli():
    uvicorn.run(
        "vllm_serve.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    cli()
