"""
Thin wrappers around whichever LLM / embedding provider is configured in
config.py. Keeping this in one module means rag_pipeline.py and
guardrails.py never need to know whether they're talking to Ollama,
OpenAI, or Anthropic.
"""

import functools
import logging

from config import settings

logger = logging.getLogger("hybrid-rag-backend")


class LLMError(Exception):
    """Raised when the configured LLM cannot be reached or errors out."""


# --------------------------------------------------------------------------
# Chat completion
# --------------------------------------------------------------------------

def chat_completion(prompt: str, system: str = "", timeout: int | None = None) -> str:
    """
    Send a single-turn prompt to the configured LLM provider and return
    the plain-text response. Raises LLMError on any provider failure so
    callers can turn it into a clean HTTP error instead of a stack trace.
    """
    timeout = timeout or settings.llm_timeout_seconds
    provider = settings.llm_provider.lower()

    try:
        if provider == "ollama":
            return _ollama_chat(prompt, system, timeout)
        elif provider == "openai":
            return _openai_chat(prompt, system, timeout)
        elif provider == "anthropic":
            return _anthropic_chat(prompt, system, timeout)
        else:
            raise LLMError(
                f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
                "Use 'ollama', 'openai', or 'anthropic'."
            )
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 - we deliberately funnel everything through LLMError
        raise LLMError(f"{provider} call failed: {exc}") from exc


def _ollama_chat(prompt: str, system: str, timeout: int) -> str:
    import httpx

    payload = {
        "model": settings.ollama_model,
        "messages": (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}]
        ),
        "stream": False,
    }
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/chat", json=payload, timeout=timeout
        )
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise LLMError(
            "Could not reach Ollama. Is it running? (ollama serve / OLLAMA_BASE_URL)"
        ) from exc
    except httpx.TimeoutException as exc:
        raise TimeoutError("Ollama request timed out") from exc
    return resp.json()["message"]["content"]


def _openai_chat(prompt: str, system: str, timeout: int) -> str:
    if not settings.openai_api_key:
        raise LLMError("OPENAI_API_KEY is not set.")
    from openai import APITimeoutError, OpenAI

    client = OpenAI(api_key=settings.openai_api_key, timeout=timeout)
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model, messages=messages
        )
    except APITimeoutError as exc:
        raise TimeoutError("OpenAI request timed out") from exc
    return resp.choices[0].message.content


def _anthropic_chat(prompt: str, system: str, timeout: int) -> str:
    if not settings.anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set.")
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=timeout)
    try:
        resp = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APITimeoutError as exc:
        raise TimeoutError("Anthropic request timed out") from exc
    return "".join(block.text for block in resp.content if block.type == "text")


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _local_embedder():
    """Lazily load the local sentence-transformers model (once per process)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading local embedding model '%s'...", settings.local_embedding_model)
    return SentenceTransformer(settings.local_embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured embedding provider."""
    provider = settings.embedding_provider.lower()

    if provider == "local":
        model = _local_embedder()
        return model.encode(texts, show_progress_bar=False).tolist()

    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.embeddings.create(
            model=settings.openai_embedding_model, input=texts
        )
        return [item.embedding for item in resp.data]

    raise LLMError(
        f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. Use 'local' or 'openai'."
    )
