from pydantic import BaseModel

from vllm_serve.config import settings


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = settings.model
    messages: list[Message]
    temperature: float = settings.default_temperature
    max_tokens: int = settings.default_max_tokens
    top_p: float = 1.0
    top_k: int = -1
    stop: list[str] | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class StreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
