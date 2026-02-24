from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "SERVE_", "env_file": ".env"}

    # Model
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer: str | None = None
    max_model_len: int | None = None

    # Engine
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    quantization: str | None = None

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 256

    # Generation defaults
    default_max_tokens: int = 1024
    default_temperature: float = 0.7


settings = Settings()
