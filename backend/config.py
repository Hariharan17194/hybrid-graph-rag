"""
Centralized settings for the backend, loaded from environment variables
(populated from a .env file via python-dotenv in main.py).

This isn't one of the files you explicitly listed, but pulling all the
provider/model switches into one place makes it much easier to change
your LLM or embedding provider later without hunting through the code.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM provider: "ollama" (local, default), "openai", or "anthropic" ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    # --- Embedding provider: "local" (default, sentence-transformers), "openai" ---
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # --- Guardrails ---
    max_question_chars: int = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
    block_suspected_injection: bool = _get_bool("BLOCK_SUSPECTED_INJECTION", True)

    # --- LLM call safety ---
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))


settings = Settings()
