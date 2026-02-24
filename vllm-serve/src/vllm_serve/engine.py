from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind

from vllm_serve.config import settings

_engine: AsyncLLMEngine | None = None


async def init_engine() -> AsyncLLMEngine:
    global _engine

    engine_args = AsyncEngineArgs(
        model=settings.model,
        tokenizer=settings.tokenizer,
        max_model_len=settings.max_model_len,
        tensor_parallel_size=settings.tensor_parallel_size,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        dtype=settings.dtype,
        quantization=settings.quantization,
    )
    _engine = AsyncLLMEngine.from_engine_args(engine_args)
    return _engine


def get_engine() -> AsyncLLMEngine:
    assert _engine is not None, "Engine not initialized"
    return _engine


def get_tokenizer():
    """Return the tokenizer from the running engine."""
    return get_engine().get_tokenizer()


def shutdown_engine():
    if _engine is not None:
        _engine.shutdown()


def build_sampling_params(
    *,
    temperature: float = settings.default_temperature,
    max_tokens: int = settings.default_max_tokens,
    top_p: float = 1.0,
    top_k: int = -1,
    stop: list[str] | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stream: bool = False,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        stop=stop or [],
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        output_kind=RequestOutputKind.DELTA if stream else RequestOutputKind.CUMULATIVE,
    )
