"""Google Gemini LLM and embedding provider using LangChain."""

from __future__ import annotations

import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

DEFAULT_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.8-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


class GeminiProviderError(Exception):
    """Raised when Gemini cannot be initialized or invoked."""


def resolve_gemini_api_key(explicit_key: str | None = None) -> str:
    """Resolve Gemini API key from explicit parameter, env vars, or Streamlit secrets."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    env_key = (os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")).strip()
    if env_key:
        return env_key
    try:
        import streamlit as st

        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if k in st.secrets and str(st.secrets[k]).strip():
                return str(st.secrets[k]).strip()
    except Exception:
        pass
    return ""


def is_gemini_configured(explicit_key: str | None = None) -> bool:
    """Return True if a non-empty Gemini API key is available."""
    return bool(resolve_gemini_api_key(explicit_key))


def list_gemini_models() -> list[str]:
    """Return supported Gemini model identifiers."""
    return list(DEFAULT_GEMINI_MODELS)


def create_gemini_llm(
    model_name: str = "gemini-3.8-flash",
    api_key: str | None = None,
    temperature: float = 0.2,
) -> ChatGoogleGenerativeAI:
    """Create a configured ChatGoogleGenerativeAI instance."""
    resolved_key = resolve_gemini_api_key(api_key)
    if not resolved_key:
        raise GeminiProviderError(
            "Google Gemini API key is not configured. Add GEMINI_API_KEY in .env or enter it in Settings."
        )
    try:
        return ChatGoogleGenerativeAI(
            model=model_name.strip() or "gemini-3.8-flash",
            api_key=resolved_key,
            temperature=temperature,
            retries=2,
        )
    except Exception as exc:
        raise GeminiProviderError(f"Failed to initialize Gemini model '{model_name}': {exc}") from exc


def create_gemini_embeddings(
    model_name: str = "models/gemini-embedding-001",
    api_key: str | None = None,
) -> GoogleGenerativeAIEmbeddings:
    """Create a GoogleGenerativeAIEmbeddings instance for RAG."""
    resolved_key = resolve_gemini_api_key(api_key)
    if not resolved_key:
        raise GeminiProviderError("Gemini API key is required for Gemini embeddings.")
    try:
        return GoogleGenerativeAIEmbeddings(
            model=model_name.strip() or "models/gemini-embedding-001",
            api_key=resolved_key,
        )
    except Exception as exc:
        raise GeminiProviderError(f"Failed to initialize Gemini embeddings: {exc}") from exc


def test_gemini_connection(model_name: str = "gemini-3.8-flash", api_key: str | None = None) -> tuple[bool, str]:
    """Verify Gemini connectivity by running a lightweight prompt."""
    try:
        llm = create_gemini_llm(model_name=model_name, api_key=api_key, temperature=0.0)
        response: Any = llm.invoke("Reply with OK.")
        content = getattr(response, "content", str(response))
        if isinstance(content, list):
            text_parts = [
                str(item["text"])
                for item in content
                if isinstance(item, dict) and item.get("type") in (None, "text") and "text" in item
            ]
            text_out = " ".join(text_parts) if text_parts else str(content)
        else:
            text_out = str(content)
        return True, f"Connected to Gemini ({model_name}): {text_out.strip()[:60]}"
    except Exception as exc:
        return False, f"Gemini connection check failed: {exc}"
