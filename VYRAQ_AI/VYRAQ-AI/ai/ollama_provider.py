"""Ollama open-source local LLM and embedding provider using LangChain."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama, OllamaEmbeddings

SUGGESTED_OLLAMA_MODELS = [
    "llama3.2",
    "llama3.1",
    "mistral",
    "qwen2.5",
    "gemma2",
    "phi3",
]


class OllamaProviderError(Exception):
    """Raised when Ollama is unreachable or fails during invocation."""


@dataclass
class OllamaStatus:
    """Health status and discovered models from a local Ollama server."""

    available: bool
    base_url: str
    models: list[str]
    message: str


def check_ollama_health(base_url: str = "http://localhost:11434", timeout: float = 2.5) -> OllamaStatus:
    """Query Ollama's /api/tags endpoint to check availability and local models."""
    clean_url = (base_url or "http://localhost:11434").strip().rstrip("/")
    tags_url = f"{clean_url}/api/tags"
    req = urllib.request.Request(tags_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        raw_models = payload.get("models", [])
        names = [str(m.get("name", "")).strip() for m in raw_models if m.get("name")]
        if names:
            return OllamaStatus(
                available=True,
                base_url=clean_url,
                models=names,
                message=f"Ollama running at {clean_url} ({len(names)} local model(s) found).",
            )
        return OllamaStatus(
            available=True,
            base_url=clean_url,
            models=[],
            message=f"Ollama is running at {clean_url}, but no models are pulled yet. Run `ollama pull llama3.2`.",
        )
    except urllib.error.URLError as exc:
        return OllamaStatus(
            available=False,
            base_url=clean_url,
            models=[],
            message=f"Ollama server unreachable at {clean_url}: {exc.reason}",
        )
    except Exception as exc:
        return OllamaStatus(
            available=False,
            base_url=clean_url,
            models=[],
            message=f"Unable to reach Ollama at {clean_url}: {exc}",
        )


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Return available local Ollama models, or suggested defaults if server is offline."""
    status = check_ollama_health(base_url=base_url)
    return status.models if status.models else []


def create_ollama_llm(
    model_name: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.2,
    verify_server: bool = True,
) -> ChatOllama:
    """Create a ChatOllama instance, checking server availability first when enabled."""
    clean_url = (base_url or "http://localhost:11434").strip().rstrip("/")
    selected_model = (model_name or "llama3.2").strip()

    if verify_server:
        status = check_ollama_health(base_url=clean_url)
        if not status.available:
            raise OllamaProviderError(
                f"Ollama server is not available at {clean_url}. Start Ollama (`ollama serve`) or switch to Gemini in Settings."
            )
        if status.models and selected_model not in status.models:
            # Allow matching base name without :latest tag
            base_matches = [m for m in status.models if m.split(":")[0] == selected_model.split(":")[0]]
            if not base_matches:
                raise OllamaProviderError(
                    f"Model '{selected_model}' is not pulled in Ollama. Available models: {', '.join(status.models)}"
                )

    try:
        return ChatOllama(
            model=selected_model,
            base_url=clean_url,
            temperature=temperature,
        )
    except Exception as exc:
        raise OllamaProviderError(f"Failed to initialize Ollama model '{selected_model}': {exc}") from exc


def create_ollama_embeddings(
    model_name: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> OllamaEmbeddings:
    """Create an OllamaEmbeddings instance for RAG."""
    clean_url = (base_url or "http://localhost:11434").strip().rstrip("/")
    return OllamaEmbeddings(model=model_name.strip() or "nomic-embed-text", base_url=clean_url)


def test_ollama_connection(model_name: str = "llama3.2", base_url: str = "http://localhost:11434") -> tuple[bool, str]:
    """Verify Ollama connectivity and model inference."""
    try:
        llm = create_ollama_llm(model_name=model_name, base_url=base_url, temperature=0.0, verify_server=True)
        response: Any = llm.invoke("Reply with OK.")
        content = getattr(response, "content", str(response))
        return True, f"Connected to Ollama ({model_name}): {str(content).strip()[:60]}"
    except Exception as exc:
        return False, str(exc)
