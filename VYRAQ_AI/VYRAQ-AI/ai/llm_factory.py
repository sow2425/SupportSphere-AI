"""Unified LLM factory and structured JSON invocation helpers."""

from __future__ import annotations

import json
import re
import time
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from ai.gemini_provider import GeminiProviderError, create_gemini_llm
from ai.groq_provider import GroqProviderError, create_groq_llm
from ai.ollama_provider import OllamaProviderError, create_ollama_llm
from config import get_config

T = TypeVar("T", bound=BaseModel)


class LLMFactoryError(Exception):
    """Raised when an LLM provider cannot be created or returns unusable output."""


def get_llm(
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    ollama_base_url: str | None = None,
    temperature: float = 0.2,
    verify_ollama: bool = True,
) -> BaseChatModel:
    """Instantiate a LangChain chat model for 'groq', 'gemini', or 'ollama'."""
    cfg = get_config()
    chosen_provider = (provider or cfg.default_provider).strip().lower()

    if chosen_provider == "groq":
        chosen_model = (model_name or cfg.groq_model).strip()
        try:
            return create_groq_llm(
                model_name=chosen_model,
                api_key=api_key or cfg.groq_api_key,
                temperature=temperature,
            )
        except GroqProviderError as exc:
            raise LLMFactoryError(str(exc)) from exc

    if chosen_provider == "gemini":
        chosen_model = (model_name or cfg.gemini_model).strip()
        try:
            return create_gemini_llm(
                model_name=chosen_model,
                api_key=api_key or cfg.gemini_api_key,
                temperature=temperature,
            )
        except GeminiProviderError as exc:
            raise LLMFactoryError(str(exc)) from exc

    if chosen_provider == "ollama":
        chosen_model = (model_name or cfg.ollama_model).strip()
        chosen_url = (ollama_base_url or cfg.ollama_base_url).strip()
        try:
            return create_ollama_llm(
                model_name=chosen_model,
                base_url=chosen_url,
                temperature=temperature,
                verify_server=verify_ollama,
            )
        except OllamaProviderError as exc:
            raise LLMFactoryError(str(exc)) from exc

    raise LLMFactoryError(
        f"Unsupported LLM provider '{chosen_provider}'. Choose 'groq', 'gemini', or 'ollama'."
    )


def _extract_message_text(response: Any) -> str:
    """Normalize LangChain AIMessage or string responses to plain text, skipping non-text blocks."""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                block_type = item.get("type")
                if block_type in (None, "text") and "text" in item:
                    parts.append(str(item["text"]))
        if parts:
            return "\n".join(parts)
    return str(content)


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from raw LLM text, stripping thinking tags and fences."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Try direct parse first, then JSONDecoder scan from each '{'
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch == "{":
            try:
                candidate, _ = decoder.raw_decode(cleaned[idx:])
                if isinstance(candidate, dict):
                    return candidate
            except json.JSONDecodeError:
                continue

    # Last attempt: strip trailing commas before } or ]
    no_trailing = re.sub(r",\s*([}\]])", r"\1", cleaned)
    first_brace = no_trailing.find("{")
    last_brace = no_trailing.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        no_trailing = no_trailing[first_brace : last_brace + 1]
    try:
        parsed = json.loads(no_trailing)
    except json.JSONDecodeError as exc:
        raise LLMFactoryError(f"Model response was not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LLMFactoryError("Expected JSON object at the top level of model response.")
    return parsed


def invoke_structured_json(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    schema_cls: type[T],
) -> T:
    """Invoke a LangChain chat model and validate its JSON output against a Pydantic schema."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    last_exc: Exception | None = None
    raw_response: Any = None
    for attempt in range(3):
        try:
            raw_response = llm.invoke(messages)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            err_msg = str(exc)
            if ("503" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise LLMFactoryError(f"LLM invocation failed: {exc}") from exc
    if last_exc is not None:
        raise LLMFactoryError(f"LLM invocation failed: {last_exc}") from last_exc

    raw_text = _extract_message_text(raw_response)
    payload = extract_json_payload(raw_text)
    # Coerce accidental null values for collection/string fields when open-source/cloud LLMs emit null
    cleaned_payload = {k: v for k, v in payload.items() if v is not None}
    try:
        return schema_cls.model_validate(cleaned_payload)
    except ValidationError as exc:
        raise LLMFactoryError(f"LLM JSON output did not match schema {schema_cls.__name__}: {exc}") from exc
