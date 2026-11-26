"""Groq LLM provider using LangChain (ChatGroq with built-in httpx BaseChatModel fallback)."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DEFAULT_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProviderError(Exception):
    """Raised when Groq cannot be initialized or invoked."""


def resolve_groq_api_key(explicit_key: str | None = None) -> str:
    """Resolve Groq API key from explicit parameter, env vars, or Streamlit secrets."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets and str(st.secrets["GROQ_API_KEY"]).strip():
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return ""


def is_groq_configured(explicit_key: str | None = None) -> bool:
    """Return True if a non-empty Groq API key is available."""
    return bool(resolve_groq_api_key(explicit_key))


def list_groq_models() -> list[str]:
    """Return supported Groq model identifiers."""
    return list(DEFAULT_GROQ_MODELS)


class GroqChatModel(BaseChatModel):
    """Lightweight LangChain BaseChatModel for Groq's OpenAI-compatible REST API."""

    model: str = "openai/gpt-oss-120b"
    api_key: str = ""
    temperature: float = 0.2
    timeout: float = 60.0

    @property
    def _llm_type(self) -> str:
        return "groq-rest-chat"

    def _convert_message(self, msg: BaseMessage) -> dict[str, str]:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "user"
        return {"role": role, "content": str(msg.content)}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(m) for m in messages],
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(GROQ_API_URL, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise GroqProviderError(f"Groq API error {resp.status_code}: {resp.text[:400]}")
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except GroqProviderError:
            raise
        except Exception as exc:
            raise GroqProviderError(f"Groq request failed: {exc}") from exc

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(text)))])


def create_groq_llm(
    model_name: str = "openai/gpt-oss-120b",
    api_key: str | None = None,
    temperature: float = 0.2,
) -> BaseChatModel:
    """Create a LangChain Groq chat model (uses ChatGroq if installed, else GroqChatModel)."""
    resolved_key = resolve_groq_api_key(api_key)
    if not resolved_key:
        raise GroqProviderError(
            "Groq API key is not configured. Add GROQ_API_KEY in .env or enter it in Settings."
        )
    chosen_model = (model_name or "openai/gpt-oss-120b").strip()

    try:
        from langchain_groq import ChatGroq  # type: ignore

        return ChatGroq(
            model=chosen_model,
            api_key=resolved_key,
            temperature=temperature,
            max_retries=2,
        )
    except ImportError:
        return GroqChatModel(
            model=chosen_model,
            api_key=resolved_key,
            temperature=temperature,
        )
    except Exception as exc:
        raise GroqProviderError(f"Failed to initialize Groq model '{chosen_model}': {exc}") from exc


def test_groq_connection(
    model_name: str = "openai/gpt-oss-120b",
    api_key: str | None = None,
) -> tuple[bool, str]:
    """Verify Groq connectivity by running a lightweight prompt."""
    try:
        llm = create_groq_llm(model_name=model_name, api_key=api_key, temperature=0.0)
        response: Any = llm.invoke("Reply with OK.")
        content = getattr(response, "content", str(response))
        return True, f"Connected to Groq ({model_name}): {str(content).strip()[:60]}"
    except Exception as exc:
        return False, f"Groq connection check failed: {exc}"
