from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PELES_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    app_name: str = "PELES Response Ranker"
    api_version: str = "1.0.0"
    cors_origins: list[str] = ["*"]

    model_backend: Literal["auto", "gemma", "heuristic"] = "auto"

    base_checkpoint: str = "unsloth/gemma-2-9b-it-bnb-4bit"
    lora_adapter_path: str | None = None
    max_sequence_length: int = 1024
    num_labels: int = 3
    inference_batch_size: int = 4

    enable_response_cache: bool = True
    response_cache_size: int = 512

    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
