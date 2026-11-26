"""AI integration and workflow package exports."""

from ai.gemini_provider import (
    DEFAULT_GEMINI_MODELS,
    GeminiProviderError,
    create_gemini_embeddings,
    create_gemini_llm,
    is_gemini_configured,
    list_gemini_models,
    test_gemini_connection,
)
from ai.groq_provider import (
    DEFAULT_GROQ_MODELS,
    GroqChatModel,
    GroqProviderError,
    create_groq_llm,
    is_groq_configured,
    list_groq_models,
    test_groq_connection,
)
from ai.llm_factory import LLMFactoryError, get_llm, invoke_structured_json
from ai.ollama_provider import (
    SUGGESTED_OLLAMA_MODELS,
    OllamaProviderError,
    OllamaStatus,
    check_ollama_health,
    create_ollama_embeddings,
    create_ollama_llm,
    list_ollama_models,
    test_ollama_connection,
)


def __getattr__(name: str):
    if name in ("build_analysis_workflow", "run_compatibility_workflow"):
        from ai import langgraph_workflow

        return getattr(langgraph_workflow, name)
    raise AttributeError(f"module 'ai' has no attribute {name!r}")


__all__ = [
    "DEFAULT_GEMINI_MODELS",
    "DEFAULT_GROQ_MODELS",
    "GeminiProviderError",
    "GroqChatModel",
    "GroqProviderError",
    "LLMFactoryError",
    "OllamaProviderError",
    "OllamaStatus",
    "SUGGESTED_OLLAMA_MODELS",
    "build_analysis_workflow",
    "check_ollama_health",
    "create_gemini_embeddings",
    "create_gemini_llm",
    "create_groq_llm",
    "create_ollama_embeddings",
    "create_ollama_llm",
    "get_llm",
    "invoke_structured_json",
    "is_gemini_configured",
    "is_groq_configured",
    "list_gemini_models",
    "list_groq_models",
    "list_ollama_models",
    "run_compatibility_workflow",
    "test_gemini_connection",
    "test_groq_connection",
    "test_ollama_connection",
]
